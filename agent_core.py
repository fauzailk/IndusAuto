import requests
from langdetect import detect
from googletrans import Translator
import os
import subprocess
import asyncio
import json

# === Remote API endpoints ===
MAIN_MODEL_URL = "http://192.168.137.180:8080/chat"  # LoRA 1 (Main Intent Model)
CONFIRMATION_MODEL_URL = "http://192.168.137.180:8081/chat"  # LoRA 2 (Confirmation Model)

translator = Translator()
conversation_history = []

# === Alpaca prompt template (matching your training format) ===
alpaca_prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def extract_assistant_response(full_text):
    """Extract the response from the Alpaca format output"""
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[1].strip()
        return response
    
    # Fallback for other formats
    if '<|eot_id|>' in full_text:
        parts = full_text.split('<|start_header_id|>assistant<|end_header_id|>')
        if len(parts) > 1:
            response = parts[-1].split('<|eot_id|>')[0].strip()
            return response
    
    return full_text.strip()

def call_remote_model(url, prompt, max_tokens=256, temperature=0.2, use_message_key=True):
    """Call remote model API"""
    try:
        if use_message_key:
            payload = {
                "message": prompt
            }
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "do_sample": True
            }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Debug print to show the exact payload being sent
        print(f"[DEBUG] Sending payload to {url}: {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"[DEBUG] Raw response: {response.text}")
        response.raise_for_status()
        
        try:
            result = response.json()
        except Exception as e:
            print(f"[ERROR] Could not parse JSON: {e}")
            return response.text
        # Assuming the API returns {"generated_text": "..."} or similar
        if "generated_text" in result:
            return result["generated_text"]
        elif "response" in result:
            return result["response"]
        elif "text" in result:
            return result["text"]
        else:
            # If the response format is different, return the full response
            return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API call failed for {url}: {e}")
        return f"[API ERROR] Failed to get response from model: {str(e)}"
    except Exception as e:
        print(f"[ERROR] Unexpected error calling {url}: {e}")
        return f"[ERROR] Unexpected error: {str(e)}"

def get_intent_prediction(user_input):
    """Get intent prediction using LoRA 1 via remote API"""
    # Send only the user input as message
    print(f"[DEBUG] LoRA 1 Input: {user_input}")
    full_response = call_remote_model(MAIN_MODEL_URL, user_input, use_message_key=True)
    print(f"[DEBUG] LoRA 1 Full output: {full_response}")
    func_key = extract_assistant_response(full_response).strip()
    print(f"[DEBUG] LoRA 1 Extracted function key: {func_key}")
    return func_key


def confirmation_llm_model(user_reply):
    """Call LoRA 2 (confirmation model) and return its response as-is."""
    # Send only the user reply as message
    print(f"[DEBUG] LoRA 2 Input: {user_reply}")
    full_response = call_remote_model(CONFIRMATION_MODEL_URL, user_reply, use_message_key=True)
    print(f"[DEBUG] LoRA 2 Full output: {full_response}")
    conf_result = extract_assistant_response(full_response).strip().lower()
    return conf_result

from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ['hi', 'bho']

def detect_language(text):
    try:
        langs = detect_langs(text)
        top_lang = langs[0].lang
        print(f"[DEBUG] Detected language: {top_lang} | Probabilities: {langs}")
        if top_lang in SUPPORTED_LANGUAGES:
            return top_lang
        else:
            return 'hi'  # fallback to Hindi if unsupported
    except Exception as e:
        print(f"[WARN] Language detection failed: {e}")
        return 'hi'

def translate_to_user_lang(text, user_lang):
    if user_lang not in ['hi', 'bho']:  # only allow Hindi or Bhojpuri
        user_lang = 'hi'
    try:
        print(f"[DEBUG] Translating to: {user_lang}")
        return translator.translate(text, dest=user_lang).text
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return text

# === Real Function Implementations ===
def AC_ON(state, user_lang):
    return translate_to_user_lang("AC चालू किया जा रहा है...", user_lang)

def AC_OFF(state, user_lang):
    return translate_to_user_lang("AC बंद किया जा रहा है...", user_lang)

import requests

def generate_osm_navigation_link(dest_lat, dest_lon, dest_name, user_lat=None, user_lon=None):
    base_url = "https://www.openstreetmap.org/directions"
    if user_lat is not None and user_lon is not None:
        # Directions from user location to destination
        return f"{base_url}?engine=fossgis_osrm_car&route={user_lat}%2C{user_lon}%3B{dest_lat}%2C{dest_lon}#map=15/{dest_lat}/{dest_lon}"
    else:
        # Just a marker to the destination
        return f"https://www.openstreetmap.org/?mlat={dest_lat}&mlon={dest_lon}#map=18/{dest_lat}/{dest_lon}"

def Search_Restaurant(state, user_lang):
    if not state.get('location'):
        state['stage'] = 'awaiting_info'
        state['info_type'] = 'location'
        return translate_to_user_lang("कृपया शहर या स्थान का नाम बताएं", user_lang)
    
    location = state['location']
    geocode_url = "https://nominatim.openstreetmap.org/search"
    geocode_params = {"q": location, "format": "json", "limit": 1}
    
    try:
        geocode_resp = requests.get(geocode_url, params=geocode_params, headers={"User-Agent": "IndusAgent/1.0"})
        geocode_data = geocode_resp.json()
        if not geocode_data:
            return translate_to_user_lang(f"{location} के लिए कोई स्थान नहीं मिला।", user_lang)

        lat = float(geocode_data[0]["lat"])
        lon = float(geocode_data[0]["lon"])
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f'''
        [out:json][timeout:25];
        (
          node["amenity"="restaurant"](around:5000,{lat},{lon});
        );
        out center 10;
        '''
        overpass_resp = requests.post(overpass_url, data=query, headers={"User-Agent": "IndusAgent/1.0"})
        data = overpass_resp.json()
        if not data.get("elements"):
            return translate_to_user_lang(f"{location} के पास कोई रेस्टोरेंट नहीं मिला।", user_lang)

        results = []
        options = []
        for el in data["elements"][:5]:
            name = el.get("tags", {}).get("name")
            rlat = el.get("lat")
            rlon = el.get("lon")
            if name and rlat and rlon:
                translated_name = translate_to_user_lang(name, user_lang)
                nav_link = generate_osm_navigation_link(rlat, rlon, name)
                results.append(translated_name)
                options.append({
                    'name': translated_name,
                    'orig_name': name,
                    'lat': rlat,
                    'lon': rlon,
                    'link': nav_link,
                })
            elif name:
                translated_name = translate_to_user_lang(name, user_lang)
                results.append(translated_name)
                options.append({
                    'name': translated_name,
                    'orig_name': name,
                    'lat': None,
                    'lon': None,
                    'link': None,
                })

        prefix = translate_to_user_lang(f"{location} के पास रेस्टोरेंट्स:", user_lang)
        if results:
            result_string = prefix + "\n" + "\n".join(f"- {r}" for r in results)
            # Store options in state for next step
            state['last_restaurants'] = options
            state['stage'] = 'awaiting_navigation_choice'
            # RETURN JUST ONCE: list and prompt
            return result_string + "\n\n" + translate_to_user_lang("सूची में से किस रेस्टोरेंट में जाना है?", user_lang)
        else:
            return translate_to_user_lang(f"{location} के पास कोई रेस्टोरेंट नहीं मिला।", user_lang)
    except Exception as e:
        return translate_to_user_lang(f"त्रुटि: {e}", user_lang)

def Search_Sites(state, user_lang):
    if not state.get('location'):
        state['stage'] = 'awaiting_info'
        state['info_type'] = 'location'
        return translate_to_user_lang("कृपया शहर या स्थान का नाम बताएं", user_lang)
    
    location = state['location']
    geocode_url = "https://nominatim.openstreetmap.org/search"
    geocode_params = {"q": location, "format": "json", "limit": 1}
    
    try:
        geocode_resp = requests.get(geocode_url, params=geocode_params, headers={"User-Agent": "IndusAgent/1.0"})
        geocode_data = geocode_resp.json()
        if not geocode_data:
            return translate_to_user_lang(f"No location found: {location}", user_lang)

        lat = float(geocode_data[0]["lat"])
        lon = float(geocode_data[0]["lon"])
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f'''
        [out:json][timeout:25];
        (
          node["tourism"="attraction"](around:5000,{lat},{lon});
        );
        out center 10;
        '''
        overpass_resp = requests.post(overpass_url, data=query, headers={"User-Agent": "IndusAgent/1.0"})
        data = overpass_resp.json()
        if not data.get("elements"):
            return translate_to_user_lang(f"No sites found near {location}.", user_lang)

        results = []
        for el in data["elements"][:5]:
            name = el.get("tags", {}).get("name")
            if name:
                translated_name = translate_to_user_lang(name, user_lang)
                results.append(translated_name)

        prefix = translate_to_user_lang(f"Sites near {location}:", user_lang)
        if results:
            result_string = prefix + "\n" + "\n".join(f"- {r}" for r in results)
            return result_string
        else:
            return translate_to_user_lang(f"No sites found near {location}.", user_lang)
    except Exception as e:
        return translate_to_user_lang(f"Error: {e}", user_lang)

function_dict = {
    "कएसी": AC_ON,
    "बएसी": AC_OFF,
    "कखा": Search_Restaurant,
    "कघु": Search_Sites
}

# === Confirmation prompts for each function and language ===
def get_confirmation_prompt(func_key, user_lang):
    prompts = {
        'कएसी': {
            'hi': 'क्या आप मेरे से AC चालू करवाना चाहते हैं?',
            'bho': 'का रउआ चाहत बानी कि हम AC चालू करीं?',
            'en': 'Do you want me to turn on the AC?'
        },
        'बएसी': {
            'hi': 'क्या आप मेरे से AC बंद करवाना चाहते हैं?',
            'bho': 'का रउआ चाहत बानी कि हम AC बंद करीं?',
            'en': 'Do you want me to turn off the AC?'
        },
        'कखा': {
            'hi': 'क्या आप मेरे से खाने के लिए जगहें खोजवाना चाहते हैं?',
            'bho': 'का रउआ चाहत बानी कि हम रउआ खातिर खाए के जगह खोजीं?',
            'en': 'Do you want me to search places for food?'
        },
        'कघु': {
            'hi': 'क्या आप मेरे से घूमने की जगहें खोजवाना चाहते हैं?',
            'bho': 'का रउआ चाहत बानी कि हम रउआ खातिर घूमे के जगह खोजीं?',
            'en': 'Do you want me to search places to visit?'
        }
    }
    return prompts.get(func_key, {}).get(user_lang, prompts.get(func_key, {}).get('hi', 'क्या आप चाहते हैं कि मैं यह कार्य करूं?'))

# === Temperature-based AC invocation logic ===
OPENWEATHERMAP_API_KEY = "7ddf545d5cb22b3616603f552e7ebe1b"  # <-- Set your API key here
CITY_NAME = "Pune"  # Default city, can be changed
TEMP_THRESHOLD = 5  # Celsius, set your threshold for AC ON
ac_state = {'on': False}

def get_current_temperature(city, api_key):
    url = (
        f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    )
    try:
        resp = requests.get(url)
        data = resp.json()
        if resp.status_code == 200 and "main" in data:
            return data["main"]["temp"]
        else:
            return None
    except Exception:
        return None

def temperature_based_ac_logic(state, user_lang):
    """Temperature-based AC logic - FROZEN for testing"""
    # Return None to skip temperature checking for now
    return None

def agentic_conversation_step(user_input, state=None):
    # === Initial state setup ===
    if state is not None and state.get("stage") == "completed" and user_input.strip():
        state = {"stage": "initial", "history": [], "ac_prompted": state.get("ac_prompted", True)}
    if state is None:
        state = {"stage": "initial", "history": [], "ac_prompted": False}

    user_lang = detect_language(user_input)
    state["lang"] = user_lang
    state["history"].append({"role": "user", "text": user_input})

    # === AC logic (temperature-based) ===
    if state["stage"] == "initial":
        temp_msg = temperature_based_ac_logic(state, user_lang)
        if temp_msg:
            return {"reply": temp_msg, "state": state, "lang": user_lang}

    # === Main model prediction ===
    if state["stage"] == "initial":
        func_key = get_intent_prediction(user_input)
        if func_key not in function_dict:
            fallback = translate_to_user_lang("माफ़ कीजिए, मैं आपकी इस अनुरोध को समझने या पूरा करने में असमर्थ हूँ।", user_lang)
            state["stage"] = "completed"
            state["history"].append({"role": "agent", "text": fallback})
            return {"reply": fallback, "state": state, "lang": user_lang}
        state["func_key"] = func_key
        state["stage"] = "awaiting_confirmation"
        prompt = get_confirmation_prompt(func_key, user_lang)
        state["history"].append({"role": "agent", "text": prompt})
        return {"reply": prompt, "state": state, "lang": user_lang}

    # === Confirm and execute ===
    elif state["stage"] == "awaiting_confirmation":
        conf_result = confirmation_llm_model(user_input)
        if conf_result == "yes":
            func_key = state.get("func_key")
            func = function_dict.get(func_key, lambda s, l: "No matching function found.")
            result = func(state, user_lang)
            state["history"].append({"role": "agent", "text": result})
            if func_key == "कएसी":
                ac_state['on'] = True
            elif func_key == "बएसी":
                ac_state['on'] = False

            if state.get("stage") == "awaiting_info":
                return {"reply": result, "state": state, "lang": user_lang}
            elif func_key == "कखा" and state.get("last_restaurants"):
                state["stage"] = "awaiting_navigation_choice"
                return {"reply": result, "state": state, "lang": user_lang}
            else:
                state["stage"] = "completed"
                return {"reply": result, "state": state, "lang": user_lang}
        else:
            reply = translate_to_user_lang("ठीक है।", user_lang)
            state["history"].append({"role": "agent", "text": reply})
            state["stage"] = "completed"
            return {"reply": reply, "state": state, "lang": user_lang}

    # === Ask for missing info ===
    elif state["stage"] == "awaiting_info":
        info_type = state.get("info_type")
        if info_type == "location":
            state["location"] = user_input
            func_key = state.get("func_key")
            func = function_dict.get(func_key, lambda s, l: "No matching function found.")
            result = func(state, user_lang)
            state["history"].append({"role": "agent", "text": result})

            if func_key == "कखा" and state.get("last_restaurants"):
                state["stage"] = "awaiting_navigation_choice"
                return {"reply": result, "state": state, "lang": user_lang}

            state["stage"] = "completed"
            return {"reply": result, "state": state, "lang": user_lang}
        else:
            msg = translate_to_user_lang("माफ़ कीजिए, मुझे और जानकारी चाहिए।", user_lang)
            state["history"].append({"role": "agent", "text": msg})
            state["stage"] = "completed"
            return {"reply": msg, "state": state, "lang": user_lang}

    # === Handle restaurant selection ===
    elif state["stage"] == "awaiting_navigation_choice":
        user_reply = user_input.strip()
        options = state.get('last_restaurants', [])
        selected = None

        if user_reply.lower() in ["नहीं", "नही", "no", "nai", "nahi"]:
            msg = translate_to_user_lang("ठीक है।", user_lang)
            state = {"stage": "initial", "history": []}
            return {"reply": msg, "state": state, "lang": user_lang}

        for option in options:
            if option['name'] in user_reply or option['orig_name'] in user_reply:
                selected = option
                break

        if selected and selected.get('link'):
            msg = translate_to_user_lang(f"नेविगेशन शुरू किया जा रहा है: {selected['name']}", user_lang)
            state["stage"] = "completed"
            state["history"].append({"role": "agent", "text": msg})
            return {"reply": msg, "state": state, "lang": user_lang}

        if "जगह" in user_reply or "सिटी" in user_reply or len(user_reply.split()) == 1:
            state["location"] = user_reply
            reply = Search_Restaurant(state, user_lang)
            state["history"].append({"role": "agent", "text": reply})
            return {"reply": reply, "state": state, "lang": user_lang}

        fallback_msg = translate_to_user_lang("कृपया सूची से एक नाम चुनें, नया स्थान बताएँ या कहें 'नहीं'.", user_lang)
        return {"reply": fallback_msg, "state": state, "lang": user_lang}

    else:
        # Completed or unknown state
        reply = translate_to_user_lang("बातचीत पूरी हुई। कृपया नया प्रश्न पूछें।", user_lang)
        return {
            "reply": reply,
            "state": {"stage": "completed", "history": state.get("history", [])},
            "lang": state.get("lang", "hi")
        }