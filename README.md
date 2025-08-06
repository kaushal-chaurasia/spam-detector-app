# 📧 Spam Detection App – Intelligent Classifier with Live Notifications

An end-to-end web app built with **Streamlit** and **Machine Learning** to classify messages as **Spam** or **Not Spam**, supporting real-time alerts via **Twilio SMS** and **SendGrid Email APIs**.

This interactive app is designed for both users and developers — allowing instant prediction, model switching, custom model uploads, and real-time feedback notifications.

🔥 Whether you're testing ML models or deploying real-world feedback systems, this app has it all.

---

## 🚀 Live App

🔗 [Try the Live App on Streamlit](https://spam-detector-app.streamlit.app)  
*(Replace with your actual Streamlit link)*

---

## 🎥 Demo

![Demo](screenshots/demo.gif)  
*(Add your demo GIF inside a `/screenshots` folder)*

---

## ✨ Features

| Feature                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| 🧠 Multiple ML Models  | Logistic Regression, Random Forest, XGBoost    |
| 📥 Upload Model        | Upload your own `.pkl` model                   |
| 🧾 CSV Upload          | Upload CSV file for batch spam prediction      |
| 🖥️ Multi-Page UI      | Navigation for Home, About, Feedback           |
| ✨ Real-Time Prediction | Type text and see instant result               |
| 📬 Email Notification  | User gets confirmation email after feedback    |
| 📲 SMS Notification    | User gets confirmation SMS via Twilio          |
| 📝 Feedback Form       | Collect feedback and save to Google Sheets     |
| 🔐 Secrets Management  | All API keys securely stored in `secrets.toml` |
| 📈 Accuracy Metrics    | Model evaluation shown with confusion matrix   |
| 🌐 Deployed Live       | Hosted using Streamlit Cloud for public access |

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python, Scikit-learn, XGBoost
- **APIs**: Twilio (SMS), SendGrid (Email)
- **Deployment**: Streamlit Cloud
- **Storage**: Google Sheets API (via `gspread` / `streamlit_gsheets`)
- **Secrets**: Managed via `.streamlit/secrets.toml`

---

## 📂 Project Structure
spam-detector-app/
├── app/
│ ├── spamdetection.py
│ ├── feedback.py
│ ├── about.py
│ └── upload_model.py
├── models/
│ ├── model.pkl
│ ├── vectorizer.pkl
│ └── ...
├── screenshots/
│ └── demo.gif
├── .streamlit/
│ └── secrets.toml
├── requirements.txt
├── README.md
└── LICENSE

🙋‍♂️ How to Use
Visit the Live App

Enter a message → choose model → click Predict

Submit feedback (email + phone number)

Receive real-time Email & SMS notification ✉️📱
-------------------------------------------------------
📄 License
This project is licensed under the MIT License — feel free to use, modify, and share with credit.

yaml
Copy
Edit
MIT License © 2025 Kaushal Chaurasia

🙌 Author
Kaushal Chaurasia

💼 LinkedIn:https://www.linkedin.com/in/kaushal-chaurasia-b6a609233/
🌐 GitHub Profile :https://github.com/kaushal-chaurasia

## ⚠️ Known Issues / Limitations

- 🧪 **Twilio SMS Trial**: The app can only send SMS to verified numbers due to Twilio trial limitations. To allow sending SMS to any number, upgrade to a paid Twilio account.
- 📧 **SendGrid Verification**: SendGrid emails will only work if the sender email is verified. For production, domain-level authentication is recommended.
- 📊 **Model Scope**: Models trained on basic SMS datasets may not generalize well for advanced or multilingual spam.
- 🚫 **Secrets File**: Do not upload `.streamlit/secrets.toml` to GitHub — it contains private API keys.

## 🧠 Train Your Own Spam Model

Want to use your custom model instead?

1. Train it using any ML pipeline (e.g., TfidfVectorizer + XGBoost)
2. Export using `joblib` or `pickle`:
```python
import joblib
joblib.dump(model, 'custom_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
---
### 📦 Twilio Trial Limitations & Upgrade

```markdown
## 🚫 Twilio Trial Limitations

By default, Twilio trial accounts can only send SMS to **verified numbers**. If an unverified number is entered, SMS will fail with a 400 error.

🔓 To remove this limitation:

1. Log into [Twilio Dashboard](https://www.twilio.com/console)
2. Click **"Upgrade Account"** (top right)
3. Add billing info
4. Purchase a Twilio phone number
5. Now your app can send SMS to any user globally!

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
- Fork the repository
- Create a new branch (`git checkout -b feature-branch`)
- Make your changes
- Commit your changes (`git commit -m 'Add some feature'`)
- Push to the branch (`git push origin feature-branch`)
- Open a Pull Request

Please make sure to follow the contribution guidelines.

📄 See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
# Contributing Guidelines

Thank you for considering contributing to this project!

## How to Contribute

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/your-repo-name.git`
3. Create a new branch: `git checkout -b feature-name`
4. Make your changes and commit them: `git commit -m 'Add new feature'`
5. Push your branch: `git push origin feature-name`
6. Submit a pull request.

Please ensure your code follows best practices and is properly documented.


