# AI Detect + AI Remove (ZIP)


- Uploads a **ZIP** containing student files (`.pdf` / `.docx`)
- Shows **AI Generated %** for each file
- Generates an **AI-removed / humanized** version (paraphrased) and shows the **new AI %**

# Run

From this folder:

```bash
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5002/`


- “AI Remove” here means **paraphrasing / rewriting** using T5 (`t5-small` by default to avoid memory issues).

