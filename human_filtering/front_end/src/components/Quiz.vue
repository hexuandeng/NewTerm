<script setup lang="ts">
const json = {
    title: "User Information",
    showProgressBar: "none",
    firstPageIsStarted: true,
    startSurveyText: "Start Quiz",
    completeText: "Start Quiz",
    completedHtml: 'Loading Questions, Please wait...',
    pages: [{
      elements: [{
        type: "html",
        html: "Welcome to our platform!<br><br><br>Your username serves as a unique identifier, ensuring that each user can fill out only one questionnaire.<br><br>The number of questions available to you is flexible, allowing you to decide on the total amount you wish to answer, up to a maximum of 900.<br><br>We also offer a feature \"Loading History\" that enabling you to load and modify your previous history.<br>If you choose \"Yes\", all the questions that you have answered before will be included in your total question count.<br>If you choose \"No\", new questions will be provided to you.<br><br><br>Enjoy your time exploring our platform!"
      },
      {
        type: "text",
        name: "Username",
        isRequired: true
      },
      {
        type: "text",
        name: "Num. of Questions",
        isRequired: true
      },
      {
        type: "radiogroup",
        name: "Loading History?",
        title: "Loading History?",
        choices: ["Yes", "No"],
        isRequired: true
      },
    ]
    }]
  };

import 'survey-core/defaultV2.min.css';
import { Model } from 'survey-core';
import axios from 'axios';
import { ref } from 'vue';
import { Converter } from "showdown";

const axiosInstance = axios.create({
  headers: {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*', // Allow any origin to access the API
  },
});

const renderComponent = ref(true);
const username = ref("")
const converter = new Converter();

const saveResults = (sender: any) => {
  const path = 'http://0.0.0.0:5000/save-json';
  const results = JSON.stringify({"username": JSON.stringify(username.value), "data": sender.data});
  axiosInstance.post(path, results)
  .then(response => {
    console.log(response.data);
  })
  .catch(error => {
    console.error("Error sending data:", error);
  });
}

const markDown = (survey: any, options: any) => {  
  // Convert Markdown to HTML
  let str = converter.makeHtml(options.text);
  // Remove root paragraphs <p></p>
  str = str.substring(3);
  str = str.substring(0, str.length - 4);
  // Set HTML markup to render
  options.html = str;
}

const getJson = (sender: any) => {
  const path = 'http://0.0.0.0:5000/return-json';
  username.value = JSON.stringify(sender.data["Username"])
  axiosInstance.post(path, JSON.stringify({username: username.value,
    num: JSON.stringify(sender.data["Num. of Questions"]),
    load: JSON.stringify(sender.data["Loading History?"])}))
  .then(response => {
    const user_json = response.data;
    renderComponent.value=false;
    survey.value = new Model(user_json);
    survey.value.onCurrentPageChanged.add(saveResults);
    survey.value.onComplete.add(saveResults);
    survey.value.onTextMarkdown.add(markDown);
  })
  .catch(error => {
    alert(error);
  });
}

const user_info = new Model(json);
const survey = ref(new Model({}));
user_info.onComplete.add(getJson);

</script>

<template>
  <SurveyComponent :model="user_info" v-if="renderComponent" />
  <SurveyComponent :model="survey" v-if="!renderComponent" />
</template>
