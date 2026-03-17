# 🔐 Project Setup: Git + SSH (Windows, Project-Level Key)

This guide documents a **production-grade workflow** to:

* Initialize a Git repository
* Use **project-level SSH keys (not global)**
* Secure credentials
* Connect to GitHub and pull existing code

---

# 📁 1. Create Project Folder

```bat
mkdir Classical_Machine_Learning
cd Classical_Machine_Learning
```

---

# 🔑 2. Generate SSH Key (Inside Project)

```bat
mkdir .ssh
ssh-keygen -t ed25519 -C "your_email@example.com" -f .ssh/id_ed25519
```

### 🔍 Output:

* Private key → `.ssh/id_ed25519`
* Public key → `.ssh/id_ed25519.pub`

---

# 📋 3. Copy Public Key (Windows)

```bat
type .ssh\id_ed25519.pub
```

👉 Copy the full key and add it to GitHub:

* Go to GitHub → **Settings**
* **SSH and GPG Keys**
* Click **New SSH Key**
* Paste and save

---

# ⚙️ 4. Initialize Git Repository

```bat
git init
```

---

# 👤 5. Configure Git Identity

```bat
git config user.name ".... ..."
git config user.email "......@gmail.com"
```

---

# 🔗 6. Add Remote Repository

⚠️ Ensure the **correct GitHub username**

```bat
git remote add origin git@github.com:sidlihe/Classical_Machine_Learning.git
```

Verify:

```bat
git remote -v
```

---

# 🔐 7. Use Project-Level SSH Key (IMPORTANT)

Since SSH key is **inside project**, override default behavior:

```bat
set GIT_SSH_COMMAND=ssh -i .ssh/id_ed25519
```

---

# 🧪 8. Test SSH Connection

```bat
ssh -i .ssh/id_ed25519 -T git@github.com
```

### ✅ Expected:

```
Hi sidlihe! You've successfully authenticated...
```

---

# ⬇️ 9. Pull Existing Repository (README)

```bat
git branch -M main
git pull origin main --allow-unrelated-histories
```

---

# 📦 10. Create Secure Structure

```bat
mkdir secrets
notepad .gitignore
```

### Add:

```
.ssh/
secrets/
.env
```

---

# 🚀 11. First Commit & Push

```bat
git add .
git commit -m "Initial setup with project-level SSH"
git push origin main
```

---

# ⚠️ Common Issues & Fixes

## ❌ `cat` not working

✔ Use:

```bat
type .ssh\id_ed25519.pub
```

---

## ❌ `Permission denied (publickey)`

✔ Ensure:

* SSH key added to GitHub
* Using correct key (`-i .ssh/id_ed25519`)

---

## ❌ `Repository not found`

✔ Cause:

* Wrong GitHub username in remote

✔ Fix:

```bat
git remote remove origin
git remote add origin git@github.com:<correct-username>/<correct-repo>.git
```

---

## ❌ `ssh-agent` commands not working

✔ Use **PowerShell**, not CMD
✔ Or skip agent and use:

```
GIT_SSH_COMMAND
```

---

# 🧠 Key Concepts

| Component       | Purpose            |
| --------------- | ------------------ |
| SSH Key         | Authentication     |
| Git Config      | Commit identity    |
| Remote URL      | Repo location      |
| GIT_SSH_COMMAND | Forces correct key |

---

# 🔒 Security Best Practices

* Never commit `.ssh/`
* Never expose private keys
* Use `.env` for secrets
* Use `.env.example` for sharing structure

---

# 🚀 Final Result

✔ Project-level SSH (isolated)
✔ Secure Git workflow
✔ Clean repository initialization
✔ Reproducible setup

---

# 🔥 Recommended Next Steps

* Add project structure (`src/`, `tests/`)
* Setup virtual environment
* Add CI/CD (GitHub Actions)
* Integrate ML pipeline

---
