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
RUN npm install
RUN npm install react-scripts@5.0.0 -g

# 앱 실행
CMD ["npm", "start"]