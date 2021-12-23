# base image
FROM node:17.2.0

# set working directory
RUN mkdir -p /app
WORKDIR /app
ADD . /app

# `/app/node_modules/.bin`을 $PATH 에 추가
ENV PATH /app/node_modules/.bin:$PATH

# app dependencies, install 및 caching
COPY demo/package.json /app/package.json

# RUN npm cache clean --force
# RUN npm config set registry http://registry.npmjs.org/

# avoid CERT Error(proxy problem???)
# 도커 CA 인증서
RUN npm config set strict-ssl false

RUN npm install
RUN npm install react-scripts@5.0.0 -g

# 앱 실행
CMD ["npm", "start"]