# Azure Deployment Guide (The Outliers)

This folder is ready for Azure App Service deployment.

## 1) Create Azure Web App
- Go to https://portal.azure.com
- Create **Web App**
- Runtime stack: **Python 3.11**
- OS: **Linux**
- Region: choose closest to your users

## 2) Deploy Code
Use one of these:
- **GitHub Deployment Center** (recommended): connect your repo/branch.
- **Zip Deploy**: upload this folder as zip (without virtual environment).

## 3) App Settings
In **Configuration -> Application settings**, add:
- `SCM_DO_BUILD_DURING_DEPLOYMENT` = `true`

In **Configuration -> General settings -> Startup Command**, set:

```bash
bash startup.sh
```

## 4) Verify
After deployment completes, open:
- `https://<your-app-name>.azurewebsites.net`

This URL is your shareable public link.

## Notes
- Large CSV files increase deploy size/time.
- If startup fails, check **Log stream** in App Service.
- `requirements.txt` must include all dependencies.
