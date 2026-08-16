import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv  
import json

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def build_prompt(data: dict) -> str:
    name = data.get("name") or ""
    summary = data.get("summary") or ""
    skills = data.get("skills") or []
    experience = data.get("experience") or []
    projects = data.get("projects") or []

    exp_lines = "\n".join(
        f"- {e.get('title', '')} at {e.get('company', '')}: {e.get('description', '') or ''}"
        for e in experience
    ) or "- (none listed)"

    proj_lines = "\n".join(
        f"- {p.get('name', '')}: {p.get('description', '') or ''}"
        for p in projects
    ) or "- (none listed)"

    prompt = f"""Write a short, engaging portfolio bio (2-4 sentences,
under 80 words) for the following person. Keep it warm and professional.
Return plain text only - no markdown, no headers, no quotes, no follow-up.

Name: {name}
Existing summary: {summary}
Skills: {", ".join(skills) if skills else "(none listed)"}
Experience:
{exp_lines}
Projects:
{proj_lines}
"""
    return prompt

@app.route("/api/bio/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "folio-bio-service",
        "api_key_set": bool(GROQ_API_KEY),
        "model": GROQ_MODEL
    })

@app.route("/api/bio", methods=["POST"])
def generate_bio():
    if not GROQ_API_KEY:
        return jsonify({
            "error": "GROQ_API_KEY is not configured on the server.",
            "hint": "Set GROQ_API_KEY in Render environment variables"
        }), 500

    data = request.get_json(silent=True) or {}
    prompt = build_prompt(data)

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You write concise, engaging professional bios."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=30,
        )
        
        if response.status_code != 200:
            return jsonify({
                "error": f"Groq API error: {response.status_code}",
                "detail": response.text
            }), response.status_code
            
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            bio = result["choices"][0].get("message", {}).get("content", "").strip()
            
            if not bio:
                return jsonify({
                    "error": "Generated bio is empty",
                    "detail": "Groq returned an empty response"
                }), 500
                
            return jsonify({"bio": bio}), 200
        else:
            return jsonify({
                "error": "Unexpected response from Groq",
                "detail": str(result)
            }), 502

    except requests.exceptions.RequestException as exc:
        return jsonify({"error": "Failed to generate bio.", "detail": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": "Unexpected error", "detail": str(exc)}), 500

@app.route("/api/bio/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok",
        "message": "Bio service is running",
        "api_key_set": bool(GROQ_API_KEY),
        "model": GROQ_MODEL
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)