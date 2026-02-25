"""
Ollama, Pipeline, Open WebUI, Jupyter, DDGS – egy helyen.

1. Közvetlen Ollama API
2. Pipeline API (OpenAI-kompatibilis)
3. Open WebUI API (saját gép: API kulcs nélkül is használható)
4. Jupyter (notebook szerver / API)
5. DDGS – DuckDuckGo Search (webes kereső backend)

Alapértelmezett címek:
  - Pipeline API:  http://10.0.0.175:9099/v1
  - Ollama API:    http://10.0.0.78:11434
  - Jupyter:       http://10.0.0.80:8888
  - Open WebUI:   saját gép, API kulcs opcionális

Ha a "Folyamat szelepek" nem jelenik meg / nem tudod kiválasztani az Open WebUI-ban:
  - Admin Panel > Beállítások > Connections: add hozzá a Pipeline kapcsolatot
    (API URL: http://10.0.0.175:9099  – a /v1 NINCS az URL-ben itt; API key: 0p3n-w3bu! vagy üres)
  - Admin Panel > Beállítások > Pipelines fül: itt jelennek meg a pipeline-ok, szelepek (valve)
  - A csevegésnél a modellválasztóban az "External" / pipeline modellek itt jelennek meg
  - Ha továbbra sem látszik: használd Pythonból pipeline_list_models() és pipeline_chat() (lásd lent).
"""

import os

# --- Alapértelmezett API címek és kulcsok ---
PIPELINE_BASE_URL = os.environ.get("PIPELINE_BASE_URL", "http://10.0.0.175:9099/v1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://10.0.0.78:11434")

JUPYTER_URL = os.environ.get("JUPYTER_URL", "http://10.0.0.80:8888")
JUPYTER_API_KEY = os.environ.get("JUPYTER_API_KEY", "b8c2a21af7639880cc03b268edebe681ec832528eda36c0b")

# Open WebUI: saját gépen fut, API kulcs nincs (webes keresőmotor); opcionális
OPENWEBUI_BASE = os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:3000")
OPENWEBUI_API_KEY = os.environ.get("OPENWEBUI_API_KEY", "")  # üres = nincs auth

# --- 1. Közvetlen Ollama API ---

def ollama_direct_chat(model: str, prompt: str, host: str | None = None):
    """Közvetlenül az Ollama API-t hívja (pl. 10.0.0.78:11434)."""
    from ollama import chat

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        host=host if host is not None else OLLAMA_HOST,
    )
    return response.get("message", {}).get("content", "")


def ollama_list_models(host: str | None = None):
    """Ollama: listázza a telepített modelleket."""
    from ollama import list as ollama_list

    client_kw = {"host": host if host is not None else OLLAMA_HOST}
    models = ollama_list(**client_kw)
    return [m.get("name") for m in models.get("models", [])]


# --- 2. Pipeline API (pl. 10.0.0.175:9099/v1, gyakran OpenAI-kompatibilis) ---

def pipeline_chat(
    model: str,
    messages: list[dict],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
):
    """
    Pipeline API chat (pl. POST .../v1/chat/completions).
    base_url pl. http://10.0.0.175:9099/v1
    """
    import requests

    base = (base_url or PIPELINE_BASE_URL).rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(
        url,
        headers=headers,
        json={"model": model, "messages": messages},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def pipeline_list_models(base_url: str | None = None, api_key: str | None = None):
    """Pipeline API: modellek listázása (GET .../v1/models), ha támogatja."""
    import requests

    base = (base_url or PIPELINE_BASE_URL).rstrip("/")
    url = f"{base}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get(url, headers=headers or None, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("data", data.get("models", []))


# --- 3. Open WebUI API (saját gép, webes kereső – API kulcs nélkül is) ---

def openwebui_chat(
    model: str,
    messages: list[dict],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """
    Open WebUI chat completions (OpenAI-kompatibilis).
    Saját gépen fut: api_key üres is lehet (nincs auth).
    """
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    key = api_key if api_key is not None else OPENWEBUI_API_KEY
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    r = requests.post(
        f"{base}/api/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def openwebui_list_models(api_key: str | None = None, base_url: str | None = None):
    """Open WebUI: listázza az elérhető modelleket (API kulcs opcionális)."""
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    key = api_key if api_key is not None else OPENWEBUI_API_KEY
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    r = requests.get(f"{base}/api/models", headers=headers or None, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("data", data.get("models", []))


# --- Open WebUI: fájlok (RAG) – feltöltés, listázás, feldolgozás állapot ---
#
# Ha "sikeresen feltöltve de nem jelenik meg" / "nincs frissíthető szelep":
# 1) A felület nem mindig frissül automatikusan – próbáld: F5 (oldal újratöltése), vagy
#    másik menüre kattintás majd vissza (pl. Knowledge → Files).
# 2) A fájl listáját Pythonból mindig lekérheted: openwebui_list_files().
# 3) A fájl először "pending", később "completed" – add a knowledge-hoz csak completed után.
# 4) Hol jelenik meg: Knowledge / Files (vagy oldalsáv Files).

def _openwebui_headers(api_key: str | None = None, accept_json: bool = True):
    key = api_key if api_key is not None else OPENWEBUI_API_KEY
    h = {"Accept": "application/json"} if accept_json else {}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def openwebui_list_files(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    content: bool = False,
):
    """
    Open WebUI: feltöltött fájlok listája (GET /api/v1/files/).
    Így ellenőrizheted, hogy a feltöltött fájl tényleg megvan-e (és milyen id/status).
    """
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    r = requests.get(
        f"{base}/api/v1/files/",
        headers=_openwebui_headers(api_key),
        params={"content": str(content).lower()},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def openwebui_upload_file(
    file_path: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    process: bool = True,
    process_in_background: bool = True,
):
    """
    Open WebUI: fájl feltöltése (POST /api/v1/files/).
    Visszaadja a választ (id, filename, data.status pl. "pending").
    """
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    headers = _openwebui_headers(api_key)
    with open(file_path, "rb") as f:
        name = os.path.basename(file_path)
        r = requests.post(
            f"{base}/api/v1/files/",
            headers=headers,
            files={"file": (name, f)},
            params={"process": process, "process_in_background": process_in_background},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def openwebui_file_process_status(
    file_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """
    Open WebUI: fájl feldolgozási állapota (pending / completed / failed).
    GET /api/v1/files/{id}/process/status
    """
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    r = requests.get(
        f"{base}/api/v1/files/{file_id}/process/status",
        headers=_openwebui_headers(api_key),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def openwebui_wait_for_file_processing(
    file_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    poll_interval: float = 2.0,
):
    """
    Vár, amíg a fájl feldolgozása completed vagy failed lesz.
    Ha completed: visszaadja a status dict-et. Ha failed: Exception-t dob.
    """
    import time

    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        data = openwebui_file_process_status(
            file_id, api_key=api_key, base_url=base_url
        )
        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Feldolgozás sikertelen: {data.get('error', data)}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Feldolgozás {timeout}s alatt nem fejeződött be.")


def openwebui_add_file_to_knowledge(
    knowledge_id: str,
    file_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """
    Feltöltött fájl hozzáadása egy Knowledge gyűjteményhez.
    Csak completed állapot után használd (pl. openwebui_wait_for_file_processing).
    """
    import requests

    base = (base_url or OPENWEBUI_BASE).rstrip("/")
    headers = _openwebui_headers(api_key)
    headers["Content-Type"] = "application/json"
    r = requests.post(
        f"{base}/api/v1/knowledge/{knowledge_id}/file/add",
        headers=headers,
        json={"file_id": file_id},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# --- 4. Jupyter (URL + API token) ---

def jupyter_session_url(*, base_url: str | None = None, token: str | None = None):
    """Jupyter belépési URL token-nel (pl. böngészőben)."""
    base = (base_url or JUPYTER_URL).rstrip("/")
    tok = token or JUPYTER_API_KEY
    return f"{base}/?token={tok}" if tok else base


def jupyter_api_headers(*, base_url: str | None = None, token: str | None = None):
    """Header dict a Jupyter REST API híváshoz (Authorization: token <token>)."""
    tok = token or JUPYTER_API_KEY
    return {"Authorization": f"token {tok}"} if tok else {}


# --- 5. DDGS – DuckDuckGo Search (webes kereső backend) ---

def ddgs_search(query: str, max_results: int = 5):
    """
    DDGS (DuckDuckGo Search) – webes szöveges keresés.
    Backend: duckduckgo-search. Visszaadja a találatok listáját.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("pip install duckduckgo-search")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
        for r in results
    ]


# --- Folyamat (Pipeline) modellek – ha a felületen nem jelenik meg a szelep ---

def list_pipeline_models(base_url: str | None = None, api_key: str | None = None):
    """
    Pipeline API-ból listázza a modelleket. Ha az Open WebUI-ban nem jelenik meg
    a folyamat szelep / modellválasztó, ezzel megnézheted, milyen modellek
    érhetők el, és pipeline_chat(model, ...) ezekkel a nevekkel hívható.
    """
    try:
        raw = pipeline_list_models(base_url=base_url, api_key=api_key)
        if isinstance(raw, list):
            return [m.get("id", m.get("name", str(m))) if isinstance(m, dict) else str(m) for m in raw]
        return raw
    except Exception:
        return []


# --- Példa használat ---

if __name__ == "__main__":
    # 0) Folyamat szelepek: Pipeline modellek (ha a felületen nem jelenik meg / nem tudod kiválasztani)
    try:
        pl_models = list_pipeline_models()
        print("Pipeline (Folyamat) modellek:", pl_models if pl_models else "(nincs vagy nem elérhető)")
    except Exception as e:
        print("Pipeline modellek:", e)

    # 1) Pipeline API (10.0.0.175:9099/v1)
    try:
        content = pipeline_chat(
            "qwen3",  # vagy a pipeline-on elérhető modell neve
            [{"role": "user", "content": "Szia! Egy rövid üzenet."}],
        )
        print("Pipeline válasz:", (content or "(üres)")[:300])
    except Exception as e:
        print("Pipeline API:", e)

    # 2) Közvetlen Ollama (10.0.0.78:11434)
    try:
        models = ollama_list_models()
        print("Ollama modellek:", models[:5] if models else "nincs / nem elérhető")
        if models:
            reply = ollama_direct_chat(models[0], "Szia! Egy rövid üzenet.")
            print("Ollama válasz:", (reply or "(üres)")[:200])
    except Exception as e:
        print("Ollama:", e)

    # 3) Open WebUI (saját gép – API kulcs nélkül is)
    try:
        content = openwebui_chat(
            "llama3.1",  # vagy a te modell neved
            [{"role": "user", "content": "Szia! Egy rövid üzenet."}],
        )
        print("Open WebUI válasz:", (content or "(üres)")[:200])
    except Exception as e:
        print("Open WebUI:", e)

    # 3b) Open WebUI feltöltött fájlok listája (ha "sikeresen feltöltve de nem jelenik meg")
    try:
        files = openwebui_list_files()
        print("Open WebUI fájlok száma:", len(files) if isinstance(files, list) else "?")
        for f in (files if isinstance(files, list) else [])[:3]:
            fid = f.get("id", f.get("filename", "?"))
            status = (f.get("data") or {}).get("status", "-")
            print(f"  - id={fid} status={status}")
    except Exception as e:
        print("Open WebUI list_files:", e)

    # 4) Jupyter URL (böngészőben megnyitás)
    print("Jupyter URL (token-nel):", jupyter_session_url())

    # 5) DDGS – DuckDuckGo Search
    try:
        results = ddgs_search("Python Ollama", max_results=2)
        for r in results:
            print("DDGS:", r.get("title", "")[:60], "-", r.get("href", "")[:50])
    except Exception as e:
        print("DDGS:", e)
