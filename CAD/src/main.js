import Vue from 'vue'
import App from './App.vue'
import store from './store'
import vuetify from './plugins/vuetify'
const { ipcRenderer, remote } = window.require('electron')

Vue.config.productionTip = false
Vue.prototype.$invoke = ipcRenderer.invoke
Vue.prototype.$ipcRenderer = ipcRenderer
Vue.prototype.$Menu = remote.Menu

new Vue({
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
