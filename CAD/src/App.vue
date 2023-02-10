<template>
  <v-app>
    <v-system-bar app window color="grey lighten-2" class="px-0 d-flex align-center" style="z-index: 20">
      <div ref="dirbar" class="d-flex align-center ml-2" style="height: 32px">
        <span style="font-weight: 600">MedAI</span>
        <v-divider vertical class="mx-2"></v-divider>
        <span class="text-caption">Chest CT 3D</span>
      </div>
      <v-spacer ref="spacer" style="height: 32px"></v-spacer>
      <div class="d-flex align-center">
        <v-btn text small tile @click="handleWindMinimize">
          <v-icon class="ma-0">mdi-window-minimize</v-icon>
        </v-btn>
        <v-btn text small tile @click="handleWindMaximize">
          <v-icon v-if="wstate===0" class="ma-0">mdi-window-maximize</v-icon>
          <v-icon v-else>mdi-window-restore</v-icon>
        </v-btn>
        <v-btn text small tile @click="handleWindClose">
          <v-icon class="ma-0">mdi-window-close</v-icon>
        </v-btn>
      </div>
    </v-system-bar>
    <v-navigation-drawer permanent app color="grey lighten-5" floating>
      <v-card tile color="grey lighten-5">
        <div class="d-flex">
          <h5 class="px-2 py-1">CT Series</h5>
          <v-spacer></v-spacer>
          <v-btn small tile icon @click="openVolume" :ripple="false">
            <v-icon>mdi-folder-open-outline</v-icon>
          </v-btn>
        </div>
      </v-card>
      <div style="height: 30%; overflow-y: scroll; border:none;">
        <v-btn small depressed tile width="256px" style="margin-top:10px" color="grey lighten-5" @click="openVolume('Series1')"><b>CT Series1</b></v-btn>
        <v-btn small depressed tile width="256px" style="margin-top:10px" color="grey lighten-5" @click="openVolume('Series2')"><b>CT Series2</b></v-btn>
        <v-btn small depressed tile width="256px" style="margin-top:10px" color="grey lighten-5" @click="openVolume('Series3')"><b>CT Series3</b></v-btn>
      </div>
      <v-card tile color="grey lighten-5">
        <div class="d-flex">
          <h5 class="px-2 py-1">Rendering Params</h5>
          <v-spacer></v-spacer>
          <v-btn small tile text @click="resetRenderingParams" :ripple="false">
            reset
          </v-btn>
        </div>
      </v-card>
      <div class="py-2 px-4 text-body-2" style="width: 100%">
        <div class="my-1 align-center justify-start">
          <span style="width: 160px; word-break: keep-all">Projection Algorithm</span>
          <v-menu offset-y left top tile>
            <template v-slot:activator="{ on, attrs }">
              <div
                class="d-flex align-center"
                style="border: 1px solid silver; width: calc(100% - 58px); cursor: pointer"
                v-on="on" v-bind="attrs"
              ><span class="ml-2">{{methods[methodIdx]}}</span>
                <v-spacer></v-spacer>
                <v-icon size="16">mdi-chevron-down</v-icon>
              </div>
            </template>
            <v-list dense>
              <v-list-item-group :value="methodIdx" @change="val=>{methodIdx=val}" color="primary" mandatory>
                <v-list-item style="min-height: 28px" class="px-2" v-for="(item, index) in methods" :key="index">
                  <v-list-item-content class="py-0">
                    <v-list-item-title>{{item}}</v-list-item-title>
                  </v-list-item-content>
                </v-list-item>
              </v-list-item-group>
            </v-list>
          </v-menu>
        </div>
        <RangeSlider class="my-2"
          v-model="ctwindow"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="70px"
          label="Model Display Window"
        ></RangeSlider>
        <SingleSlider class="my-2"
          v-model="opacity"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="50px"
          append-type="value"
          label="Model Opacity"
        ></SingleSlider>
        <SingleSlider class="my-2"
          v-model="isothres"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="50px"
          append-type="value"
          label="Isosurface Threshold"
        ></SingleSlider>
        <SingleSlider class="my-2" v-show="methodIdx !== 2"
          v-model="low"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="50px"
          append-type="value"
          label="Low Density Threshold"
        ></SingleSlider>
        <SingleSlider class="my-2" v-show="methodIdx !== 2"
          v-model="high"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="50px"
          append-type="value"
          label="High Density Threshold"
        ></SingleSlider>
        <div class="my-1 align-center justify-start">
          <span style="width: 160px; word-break: keep-all">Model Colormap</span>
          <v-menu offset-y left top tile>
            <template v-slot:activator="{ on, attrs }">
              <div
                class="d-flex align-center"
                style="border: 1px solid silver; width: calc(100% - 58px); cursor: pointer"
                v-on="on" v-bind="attrs"
              ><span class="ml-2">{{cmNames[cmIndex]}}</span>
                <v-spacer></v-spacer>
                <v-icon size="16">mdi-chevron-down</v-icon>
              </div>
            </template>
            <v-list dense>
              <v-list-item-group :value="cmIndex" @change="val=>{cmIndex=val}" color="primary" mandatory>
                <v-list-item style="min-height: 28px" class="px-2" v-for="(item, index) in cmNames" :key="index">
                  <v-list-item-content class="py-0">
                    <v-list-item-title>{{item}}</v-list-item-title>
                  </v-list-item-content>
                </v-list-item>
              </v-list-item-group>
            </v-list>
          </v-menu>
        </div>
        <div class="mt-2">
          <div class="d-flex align-center justify-start">
            <span class="mb-1 mr-2">Lesion Highlight</span>
            <v-switch v-model="heat"></v-switch>
          </div>
        </div>
        <SingleSlider class="mb-2" v-show="heat"
          v-model="heatRate"
          :min="0" :max="100"
          height="16"
          label-width="160px"
          append-width="50px"
          append-type="value"
          label="Heatmap Intensity"
        ></SingleSlider>
        <div class="my-1 align-center justify-start" v-show="heat">
          <span style="width: 160px; word-break: keep-all">Heat Colormap</span>
          <v-menu offset-y left top tile>
            <template v-slot:activator="{ on, attrs }">
              <div
                class="d-flex align-center"
                style="border: 1px solid silver; width: calc(100% - 58px); cursor: pointer"
                v-on="on" v-bind="attrs"
              ><span class="ml-2">{{cmNames[heatIndex]}}</span>
                <v-spacer></v-spacer>
                <v-icon size="16">mdi-chevron-down</v-icon>
              </div>
            </template>
            <v-list dense>
              <v-list-item-group :value="heatIndex" @change="val=>{heatIndex=val}" color="primary" mandatory>
                <v-list-item style="min-height: 28px" class="px-2" v-for="(item, index) in cmNames" :key="index">
                  <v-list-item-content class="py-0">
                    <v-list-item-title>{{item}}</v-list-item-title>
                  </v-list-item-content>
                </v-list-item>
              </v-list-item-group>
            </v-list>
          </v-menu>
        </div>
      </div>
    </v-navigation-drawer>
    <v-navigation-drawer permanent app right color="grey lighten-5" floating>
      <v-card tile color="grey lighten-5">
        <div class="d-flex">
          <h5 class="px-2 py-1">Pathology Prediction</h5>
          <v-spacer></v-spacer>
        </div>
      </v-card>
      <div class="py-2 px-4 text-body-2" style="width: 100%">
        <div class="my-1 align-center justify-start">
          <v-simple-table>
            <tr><b>Thymoma</b></tr>
            <tr><v-progress-linear ref="p1" light></v-progress-linear></tr>
            <h4 ref="prob1"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Benign cyst</b></tr>
            <tr><v-progress-linear ref="p2"></v-progress-linear></tr>
            <h4 ref="prob2"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Thymic carcinoma</b></tr>
            <tr><v-progress-linear ref="p3" light></v-progress-linear></tr>
            <h4 ref="prob3"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Germinoma</b></tr>
            <tr><v-progress-linear ref="p4" light></v-progress-linear></tr>
            <h4 ref="prob4"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Neuroendocrine tumor</b></tr>
            <tr><v-progress-linear ref="p5" light></v-progress-linear></tr>
            <h4 ref="prob5"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Thymic hyperplasia</b></tr>
            <tr><v-progress-linear ref="p6" light></v-progress-linear></tr>
            <h4 ref="prob6"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Lymphoma</b></tr>
            <tr><v-progress-linear ref="p7" light></v-progress-linear></tr>
            <h4 ref="prob7"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Lymphadenosis</b></tr>
            <tr><v-progress-linear ref="p8" light></v-progress-linear></tr>
            <h4 ref="prob8"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Ectopicthyroidgland</b></tr>
            <tr><v-progress-linear ref="p9" light></v-progress-linear></tr>
            <h4 ref="prob9"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Granulomatous inflammation</b></tr>
            <tr><v-progress-linear ref="p10" light></v-progress-linear></tr>
            <h4 ref="prob10"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Neurogenic tumor</b></tr>
            <tr><v-progress-linear ref="p11" light></v-progress-linear></tr>
            <h4 ref="prob11"><b>score = 0</b></h4>
            <p style="margin-bottom: 10px;"></p>
            <tr><b>Other soft tissue neoplasm</b></tr>
            <tr><v-progress-linear ref="p12" light></v-progress-linear></tr>
            <h4 ref="prob12"><b>score = 0</b></h4>
          </v-simple-table>
        </div>
      </div>
    </v-navigation-drawer>
    <v-main>
      <ModelViewer ref="model"
        :cm-index="cmIndex"
        :heat-index="heatIndex"
        :heat="heat"
        :heat-rate="heatRate"
        :isothres="isothres"
        :ctwindow="ctwindow"
        :low="low"
        :high="high"
        :opacity="opacity"
        :method-idx="methodIdx"
      ></ModelViewer>
    </v-main>
    <v-footer app dark height="28px" class="text-caption d-flex pa-0" style="z-index: 20">
      <div class="px-2">
        <div v-if="hasError" class="d-flex align-center">
          <v-icon size="14" class="mr-2">mdi-alert-rhombus-outline</v-icon>
          Operation failed.
          <v-btn class="pa-0 mx-2" small text @click="errorActive=true" v-if="!errorActive">Show Detail</v-btn>
          <v-btn class="pa-0 mx-2" small text @click="errorActive=false" v-else>Close Detail</v-btn>
        </div>
        <div v-else-if="loading!==''">
          <v-progress-circular indeterminate size="14" width="2" class="mr-2"></v-progress-circular>
          {{loading}}
        </div>
        <div v-else>Ready</div>
      </div>
    </v-footer>
    <ErrorPanel
      :active="errorActive"
      :detail="errorDetail"
    ></ErrorPanel>
  </v-app>
</template>

<script>
import ModelViewer from './components/ModelViewer'
import ErrorPanel from './components/ErrorPanel'
import RangeSlider from './components/RangeSlider'
import SingleSlider from './components/SingleSlider'
export default {
  name: 'App',
  components: {
    ModelViewer,
    ErrorPanel,
    RangeSlider,
    SingleSlider
  },
  mounted () {
    this.$refs.spacer.style['-webkit-app-region'] = 'drag'
    this.$refs.dirbar.style['-webkit-app-region'] = 'drag'
    this.$ipcRenderer.on('stdOut', (event, msg) => {
      const text = new TextDecoder('utf-8').decode(msg)
      this.errorDetail += msg + '\n'
      console.log(text)
    })
    this.$ipcRenderer.on('stdErr', (event, msg) => {
      const text = new TextDecoder('utf-8').decode(msg)
      this.errorDetail += msg + '\n'
      console.log(text)
    })
    this.$ipcRenderer.on('maximize', (event) => {
      this.wstate = 1
    })
    this.$ipcRenderer.on('unmaximize', (event) => {
      this.wstate = 0
    })
    this.$ipcRenderer.on('restore', (event) => {
      this.wstate = 0
    })
  },
  data: () => ({
    errorDetail: '',
    filename: '',
    wstate: 0,
    hasError: false,
    errorActive: false,
    loading: '',

    // rendering params:
    cmIndex: 1,
    heatIndex: 0,
    heatRate: 40,
    heat: false,
    isothres: 25,
    ctwindow: [0, 100],
    low: 10,
    high: 20,
    opacity: 70,
    methods: ['Area Count', 'Pseudo X-Ray', 'Maximum Intensity'],
    methodIdx: 2,
    cmNames: ['Viridis', 'Inferno', 'Cool', 'Gray']
  }),
  methods: {
    handleWindMinimize () {
      this.$invoke('minWindow')
    },
    handleWindMaximize () {
      this.$invoke('maxWindow')
    },
    handleWindClose () {
      this.$invoke('closeWindow')
    },
    async openVolume (filename) {
      try {
        this.loading = 'Analyzing CT volume...'
        this.loading = 'Reconstructing CT volume...'
        await this.$refs.model.loadModel(filename)
        if (filename.includes('Series1')) {
          this.$refs.p1.value = 99.96
          this.$refs.p1.color = 'red'
          this.$refs.prob1.textContent = 'score = 0.9996'
          this.$refs.p2.value = 0
          this.$refs.p2.color = 'blue lighten-4'
          this.$refs.prob2.textContent = 'score = 0'
          this.$refs.p3.value = 0
          this.$refs.p3.color = 'blue lighten-4'
          this.$refs.prob3.textContent = 'score = 0'
          this.$refs.p4.value = 0
          this.$refs.p4.color = 'blue lighten-4'
          this.$refs.prob4.textContent = 'score = 0'
          this.$refs.p5.value = 0.01
          this.$refs.p5.color = 'red'
          this.$refs.prob5.textContent = 'score = 0.0001'
          this.$refs.p6.value = 0
          this.$refs.p6.color = 'blue lighten-4'
          this.$refs.prob6.textContent = 'score = 0'
          this.$refs.p7.value = 0.03
          this.$refs.p7.color = 'red'
          this.$refs.prob7.textContent = 'score = 0.0003'
          this.$refs.p8.value = 0
          this.$refs.p8.color = 'blue lighten-4'
          this.$refs.prob8.textContent = 'score = 0'
          this.$refs.p9.value = 0
          this.$refs.p9.color = 'blue lighten-4'
          this.$refs.prob9.textContent = 'score = 0'
          this.$refs.p10.value = 0
          this.$refs.p10.color = 'blue lighten-4'
          this.$refs.prob10.textContent = 'score = 0'
          this.$refs.p11.value = 0
          this.$refs.p11.color = 'blue lighten-4'
          this.$refs.prob11.textContent = 'score = 0'
          this.$refs.p12.value = 0
          this.$refs.p12.color = 'blue lighten-4'
          this.$refs.prob12.textContent = 'score = 0'
        } else if (filename.includes('Series2')) {
          this.$refs.p1.value = 0
          this.$refs.p1.color = 'blue lighten-4'
          this.$refs.prob1.textContent = 'score = 0'
          this.$refs.p2.value = 5.05
          this.$refs.p2.color = 'red'
          this.$refs.prob2.textContent = 'score = 0.0505'
          this.$refs.p3.value = 0
          this.$refs.p3.color = 'blue lighten-4'
          this.$refs.prob3.textContent = 'score = 0'
          this.$refs.p4.value = 0.97
          this.$refs.p4.color = 'red'
          this.$refs.prob4.textContent = 'score = 0.0097'
          this.$refs.p5.value = 0
          this.$refs.p5.color = 'blue lighten-4'
          this.$refs.prob5.textContent = 'score = 0'
          this.$refs.p6.value = 0
          this.$refs.p6.color = 'blue lighten-4'
          this.$refs.prob6.textContent = 'score = 0'
          this.$refs.p7.value = 0
          this.$refs.p7.color = 'blue lighten-4'
          this.$refs.prob7.textContent = 'score = 0'
          this.$refs.p8.value = 93.04
          this.$refs.p8.color = 'red'
          this.$refs.prob8.textContent = 'score = 0.9304'
          this.$refs.p9.value = 0.08
          this.$refs.p9.color = 'blue lighten-4'
          this.$refs.prob9.textContent = 'score = 0.0008'
          this.$refs.p10.value = 0.84
          this.$refs.p10.color = 'blue lighten-4'
          this.$refs.prob10.textContent = 'score = 0.0084'
          this.$refs.p11.value = 0
          this.$refs.p11.color = 'blue lighten-4'
          this.$refs.prob11.textContent = 'score = 0'
          this.$refs.p12.value = 0
          this.$refs.p12.color = 'blue lighten-4'
          this.$refs.prob12.textContent = 'score = 0'
        } else if (filename.includes('Series3')) {
          this.$refs.p1.value = 0
          this.$refs.p1.color = 'blue lighten-4'
          this.$refs.prob1.textContent = 'score = 0'
          this.$refs.p2.value = 0.01
          this.$refs.p2.color = 'red'
          this.$refs.prob2.textContent = 'score = 0.0001'
          this.$refs.p3.value = 0
          this.$refs.p3.color = 'blue lighten-4'
          this.$refs.prob3.textContent = 'score = 0'
          this.$refs.p4.value = 0
          this.$refs.p4.color = 'blue lighten-4'
          this.$refs.prob4.textContent = 'score = 0'
          this.$refs.p5.value = 0
          this.$refs.p5.color = 'blue lighten-4'
          this.$refs.prob5.textContent = 'score = 0'
          this.$refs.p6.value = 0
          this.$refs.p6.color = 'blue lighten-4'
          this.$refs.prob6.textContent = 'score = 0'
          this.$refs.p7.value = 0
          this.$refs.p7.color = 'blue lighten-4'
          this.$refs.prob7.textContent = 'score = 0'
          this.$refs.p8.value = 0
          this.$refs.p8.color = 'blue lighten-4'
          this.$refs.prob8.textContent = 'score = 0'
          this.$refs.p9.value = 0
          this.$refs.p9.color = 'blue lighten-4'
          this.$refs.prob9.textContent = 'score = 0'
          this.$refs.p10.value = 0
          this.$refs.p10.color = 'blue lighten-4'
          this.$refs.prob10.textContent = 'score = 0'
          this.$refs.p11.value = 88.18
          this.$refs.p11.color = 'red'
          this.$refs.prob11.textContent = 'score = 0.8818'
          this.$refs.p12.value = 11.81
          this.$refs.p12.color = 'red'
          this.$refs.prob12.textContent = 'score = 0.1181'
        }
      } catch (err) {
        this.hasError = true
        this.errorDetail += err + '\n'
        console.log(err)
      } finally {
        this.loading = ''
      }
    },
    resetRenderingParams () {
      this.cmIndex = 1
      this.isothres = 25
      this.ctwindow = [0, 100]
      this.low = 10
      this.high = 20
      this.opacity = 70
      this.methodIdx = 2
      this.heatIndex = 0
      this.heatRate = 40
      this.heat = false
      this.$refs.model.resetCamera()
    }
  }
}
</script>
