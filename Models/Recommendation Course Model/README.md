---
title: Course Recommendation API
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Course Recommendation API 🎓

An AI-powered course recommendation chatbot exposed as a FastAPI REST API.

Built with **TF-IDF + Random Forest** for fast, lightweight inference — no heavy model downloads.

---

## 📡 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/`         | Health check |
| GET  | `/docs`     | ✅ Swagger UI — test everything in browser |
| GET  | `/welcome`  | Random welcome message |
| GET  | `/options`  | All subjects, frameworks, levels, languages |
| POST | `/recommend`| Stateless one-shot recommendation |
| POST | `/chat`     | Stateful conversational chatbot |
| POST | `/reset`    | Clear a session |

---

## 💬 Chat Example

**POST** `/chat`
```json
{ "session_id": "user1", "message": "I want react beginner courses" }
```

**Response:**
```json
{
  "reply": "🎓 Here are your recommended courses:",
  "courses": [
    { "title": "React for Beginners", "level": "beginner level", "url": "..." }
  ],
  "step": "post_recommendation",
  "session_id": "user1"
}
```

---

## 🔁 Conversation Flow

```
You:  "I want backend courses"
Bot:  "Frameworks: nodejs, spring, .net ... Choose one:"

You:  "nodejs"
Bot:  "Levels: beginner level, expert level. Choose a level:"

You:  "beginner"
Bot:  "🎓 Here are your recommended courses: ..."

You:  "show me expert instead"
Bot:  "🎓 Updated courses: ..."

You:  "bye"
Bot:  "Bye! Happy learning! 📚"
```

---

## ⚡ One-shot Recommendation

**POST** `/recommend`
```json
{ "framework": "react", "level": "beginner level" }
```

No session needed — instant results.
