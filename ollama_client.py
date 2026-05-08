from __future__ import annotations

import argparse
import json
import re
import time
from typing import Callable

import requests

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "water-practices"
DEFAULT_TIMEOUT = 300

# ------------------------- Parsing robusto -----------------------------------

def parse_response(text: str) -> dict:
    """Parsea el JSON tolerando fences, preámbulos, etc.

    Si falla, devuelve un dict vacío con _parse_error para no romper pipelines.
    """
    if not text:
        return _empty("Empty response")

    text = text.strip()
    # Eliminar fences markdown
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            text = text.lstrip("`").lstrip("json").strip()

    fb = text.find("{")
    lb = text.rfind("}")
    if fb >= 0 and lb > fb:
        text = text[fb:lb + 1]

    try:
        result = json.loads(text)
        if "contains_practice" not in result:
            result["contains_practice"] = False
        if "practices" not in result:
            result["practices"] = []
        if "summary" not in result:
            result["summary"] = ""
        if not result["contains_practice"]:
            result["practices"] = []
        return result
    except json.JSONDecodeError as e:
        return _empty(f"Parse error: {e}")


def _empty(reason: str) -> dict:
    return {
        "contains_practice": False,
        "practices": [],
        "summary": "",
        "_parse_error": reason,
    }


# ------------------------- Cliente -------------------------------------------

class WaterPracticesClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"No se puede conectar a Ollama en {self.base_url}. "
                f"¿Está corriendo? Error: {e}"
            )

        models = [m["name"] for m in r.json().get("models", [])]
        if not any(m.startswith(self.model) for m in models):
            raise RuntimeError(
                f"Modelo '{self.model}' no encontrado en Ollama. "
                f"Modelos disponibles: {models}\n"
                f"Crea el modelo con: ollama create {self.model} -f .\\Modelfile"
            )

    # ---------- API principal ----------

    def analyze_text(self, text: str) -> dict:
        user_message = f"Analyze this chunk:\n\n---\n{text}\n---"

        try:
            r = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_message,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "num_ctx": 4096,
                        "num_predict": 1024,
                    },
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            response_text = r.json().get("response", "")
            return parse_response(response_text)
        except requests.exceptions.Timeout:
            return _empty(f"Timeout after {self.timeout}s")
        except Exception as e:
            return _empty(f"Request error: {e}")

    def analyze_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        results = []
        n = len(texts)
        for i, text in enumerate(texts):
            results.append(self.analyze_text(text))
            if progress_callback is not None:
                progress_callback(i + 1, n)
        return results


# ------------------------- CLI de validación ---------------------------------

DEMO_EXAMPLES = [
    "We deployed ESP32-based loggers powered by 5W solar panels with MPPT controllers, recording water level data offline to SD cards. Local farmers were trained to download data weekly using a smartphone app.",
    "The system requires constant 4G/5G connectivity to Azure cloud services for real-time data processing. Without internet, the entire monitoring infrastructure fails to operate. Each sensor unit costs over $5000 and requires specialized engineers for calibration.",
    "This paper presents a comprehensive review of water resources in arid regions. The authors discuss historical perspectives and policy frameworks dating back to 1850.",
    "References: Smith, J. (2020). Water quality monitoring. Journal of Hydrology, 45(2), 123-145. Doe, A. (2021). Rural sensors review.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Texto a analizar")
    parser.add_argument("--demo", action="store_true",
                        help="Correr 4 ejemplos de demostración")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    args = parser.parse_args()

    print(f"[client] Conectando a Ollama en {args.url}...")
    client = WaterPracticesClient(model=args.model, base_url=args.url)
    print(f"[client] Modelo '{args.model}' disponible. Listo.\n")

    if args.text:
        examples = [args.text]
    elif args.demo:
        examples = DEMO_EXAMPLES
    else:
        print("Usa --text 'texto' o --demo. Ejemplo:")
        print("  python -m local.ollama_client --demo")
        return

    for i, ex in enumerate(examples, 1):
        print(f"{'=' * 60}")
        print(f"EJEMPLO {i}/{len(examples)}")
        print(f"{'=' * 60}")
        print(f"INPUT: {ex[:200]}{'...' if len(ex) > 200 else ''}\n")

        t0 = time.time()
        result = client.analyze_text(ex)
        elapsed = time.time() - t0

        print(f"OUTPUT (tomó {elapsed:.1f}s):")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()