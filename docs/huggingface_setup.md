# Putting Swirl Explorer on the web (Hugging Face Space)

This is a one-time setup of about 5 minutes. After it, every push to `main` that changes the web app
updates the website automatically, through a GitHub Actions workflow in this repository.

You need: a Hugging Face account (free) and admin access to the GitHub repository
`DakotaMath/LiLiYanVisualizer`.

## 1. Create a Hugging Face account

Sign up at <https://huggingface.co/join> (free). Note your **username**; it appears in the Space's address.

## 2. Create the Space (the web server the app runs on)

1. Go to <https://huggingface.co/new-space>.
2. Fill in:
   - **Owner:** your username.
   - **Space name:** `swirl-explorer`. Any name works, but it becomes part of the address.
   - **Short description:** e.g. "Axisymmetric Navier–Stokes solutions with swirl".
   - **License:** your choice (or leave empty).
   - **Space SDK:** **Docker**, then the **Blank** template.
   - **Space hardware:** **CPU basic · 2 vCPU · 16 GB · FREE**.
   - **Public** (anyone with the link can use it) or **Private** (only you, while testing).
3. Click **Create Space**. It starts empty; leave it as it is. The GitHub workflow fills it in.

The app will be at `https://huggingface.co/spaces/<username>/swirl-explorer`.

## 3. Create an access token (lets GitHub update the Space)

1. Go to <https://huggingface.co/settings/tokens>, or avatar → **Settings → Access Tokens**.
2. Click **+ Create new token**.
3. Choose one:
   - **Write** (simplest): give it a name such as `github-deploy`.
   - **Fine-grained** (tighter): under **Repositories permissions**, pick your `swirl-explorer`
     Space and tick **Write access to contents/settings of selected repos**.
4. Click **Create token** and **copy it now**; Hugging Face shows it only once.
   Treat it like a password: don't paste it into chat, email or the code.

## 4. Give the token and the Space name to GitHub

In the GitHub repository: **Settings → Secrets and variables → Actions**.

1. **Secrets** tab → **New repository secret**:
   - Name: `HF_TOKEN`
   - Secret: the token from step 3
2. **Variables** tab → **New repository variable**:
   - Name: `HF_SPACE`
   - Value: `<username>/swirl-explorer` (your username and the Space name from step 2)

That's all. Until both exist, the deploy workflow skips the upload and says so in its log,
so nothing fails.

## What happens next

- Claude builds and tests the web version on the `dev/web` branch and opens a pull request.
- When you merge it, the **Deploy web app** workflow uploads the app to your Space. Hugging Face then
  builds the container (a few minutes the first time) and the app goes live at the address above.
- **Free-tier limits:**
  - **Sleep:** the Space sleeps after 48 hours without visitors. The next visitor waits about a
    minute while it wakes.
  - **Shared CPUs:** everyone shares the 2 CPUs, so large Σ-maps are slower than on your PC; the
    web version uses a coarser default grid.

## Undoing it

- **Take the site down:** in the Space, go to **Settings** and either set it to **Private** or
  **Delete this Space**.
- **Revoke GitHub's access:** delete the token on <https://huggingface.co/settings/tokens>.
