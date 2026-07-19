# Outcome Forecast — web app

A small Flask site that puts the trained model behind a form: enter a
student's details, get back a predicted outcome (Distinction / Pass / Fail /
Withdrawn) with a probability for each.

## Run it locally

```
cd webapp
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Files

```
webapp/
├── app.py              # Flask routes: "/" (page) and "/api/predict" (JSON API)
├── model.pkl           # the trained pipeline (copied from ../models/best_model.pkl)
├── preprocessing.py     # feature engineering used at both train and predict time
├── templates/index.html # the page (form + results, no separate JS/CSS files)
├── requirements.txt
└── Procfile             # tells Render/Railway/Heroku how to start the app
```

## Put it online (free options)

Any of these work for a small project like this. All three read the
`Procfile` and `requirements.txt` automatically — you don't need to change
any code.

### Render (recommended — simplest free option)
1. Push this `webapp/` folder to a GitHub repo.
2. Go to render.com → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `gunicorn app:app`
4. Deploy. Render gives you a public URL like `https://yourapp.onrender.com`.
   Free tier sleeps after inactivity — the first request after a while takes
   ~30s to wake up.

### Railway
1. Push to GitHub, go to railway.app → New Project → Deploy from repo.
2. Railway auto-detects the `Procfile`. No other config needed.
3. Generate a public domain from the service settings.

### Fly.io
1. Install the `flyctl` CLI, run `fly launch` inside `webapp/` — it detects
   Flask/Python automatically and writes a `fly.toml`.
2. `fly deploy`. You get a `https://yourapp.fly.dev` URL.

### If you don't want to use GitHub
Render, Railway, and Fly.io all also accept a direct folder/zip upload or a
CLI-based deploy from your machine — check each platform's "deploy without
git" docs if that's preferred.

## Notes for real use

- `app.py` currently allows any request through `/api/predict` with no
  authentication or rate limiting. Fine for a demo; add an API key check or
  login before sharing widely.
- Input validation is minimal (numeric fields just need to parse as
  numbers). For production, validate ranges too (e.g., scores 0–100).
- If you retrain the model later, just replace `model.pkl` in this folder —
  no other changes needed, as long as the feature set stays the same.
