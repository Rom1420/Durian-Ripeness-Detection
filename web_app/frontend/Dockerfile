# Étape 1 : build de l'application
FROM node:20 AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# Étape 2 : servir les fichiers statiques avec Nginx
FROM nginx:alpine

# Copier le build React dans le dossier de contenu statique de Nginx
COPY --from=build /app/build /usr/share/nginx/html

# Exposer le port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]