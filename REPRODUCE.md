1) Install environment manager (for example, `micromamba`)
2) create env using `environment.yaml` (for example, `micromamba create -f environment.yaml --channel-priority flexible`)
3) Activate the environment (for example, `micromamba activate ai-detector`)
4) Создайте аккаунт на DagsHub
5) Настройте аутентификацию. DagsHub использует специальный токен.

```
export AWS_ACCESS_KEY_ID=<your_dagshub_token>
export AWS_SECRET_ACCESS_KEY=<your_dagshub_token>
```
6) Загрузите данные и модель (`dvc pull`)
7) Воспроизведите пайплайн (`dvc repro`)