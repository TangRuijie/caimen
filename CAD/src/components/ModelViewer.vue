<template>
  <div>
    <div ref="canvas" v-resize="handleResize" style="position: fixed; top: 32px; bottom: 32px; left: 256px; right: 256px"></div>
  </div>
</template>

<script>
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { MyVolumeRenderShader } from '../plugins/MyVolumeShader'
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader'
// import path from 'path'
export default {
  name: 'ModelViewer',
  data: el => ({
    scene: null,
    camera: null,
    renderer: null,
    camHeight: 256,
    controls: null,

    model: null,
    modelX: 0,
    modelY: 0,
    modelZ: 0,

    methods: ['Area Count Projection', 'Pseudo X-Ray Projection', 'Maximum Intensity Projection'],

    cmValues: [null, null, null, null],

    texture: null,
    heatTexture: null,

    working: false
  }),
  props: {
    cmIndex: Number,
    heatIndex: Number,
    isothres: Number,
    ctwindow: Array,
    low: Number,
    high: Number,
    opacity: Number,
    methodIdx: Number,
    heat: Boolean,
    heatRate: Number
  },
  watch: {
    cmIndex: function () {
      this.updateUniform()
    },
    isothres: function () {
      this.updateUniform()
    },
    ctwindow: function () {
      this.updateUniform()
    },
    low: function () {
      this.updateUniform()
    },
    high: function () {
      this.updateUniform()
    },
    opacity: function () {
      this.updateUniform()
    },
    methodIdx: function () {
      this.updateUniform()
    },
    heatIndex: function () {
      this.updateUniform()
    },
    heat: function () {
      this.updateUniform()
    },
    heatRate: function () {
      this.updateUniform()
    }
  },
  mounted () {
    const width = window.innerWidth - 512
    const height = window.innerHeight - 60
    const aspect = width / height
    this.scene = new THREE.Scene()
    this.camera = new THREE.OrthographicCamera(
      this.camHeight * aspect, this.camHeight * aspect, this.camHeight, -this.camHeight, 1, 1000
    )
    this.renderer = new THREE.WebGLRenderer()
    this.camera.position.set(0, 0, 256)
    this.camera.up.set(0, 0, 1)
    this.camera.updateProjectionMatrix()
    this.$refs.canvas.appendChild(this.renderer.domElement)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.target.set(0, 0, 0)
    this.controls.update()
    this.cmValues[0] = new THREE.TextureLoader().load('cm_viridis.png')
    this.cmValues[1] = new THREE.TextureLoader().load('cm_inferno.png')
    this.cmValues[2] = new THREE.TextureLoader().load('cm_cool.png')
    this.cmValues[3] = new THREE.TextureLoader().load('cm_gray.png')
    this.$nextTick(() => {
      this.handleResize()
    })
    this.animate()
  },
  methods: {
    handleResize () {
      if (this.renderer) {
        const width = window.innerWidth - 512
        const height = window.innerHeight - 60
        this.renderer.setSize(width, height)
        this.camera.left = -this.camHeight * width / height
        this.camera.right = this.camHeight * width / height
        this.camera.position.set(0, 0, 256)
        this.camera.up.set(0, 0, 1)
        this.camera.updateProjectionMatrix()
        this.controls.update()
      }
    },
    resetCamera () {
      this.handleResize()
      if (this.controls) {
        this.controls.target.set(this.modelX / 2, this.modelY / 2, this.modelZ / 2)
        this.controls.update()
      }
    },
    async loadModel (name) {
      if (this.working) return
      this.working = true
      const data = await this.$invoke('openFile', './data/' + name + '.nrrd')
      const heatData = await this.$invoke('openFile', './data/' + name + '.heat.nrrd')
      const loader = new NRRDLoader()
      const volume = loader.parse(data)
      const heatLoader = new NRRDLoader()
      const heatVolume = heatLoader.parse(heatData)
      console.log(volume)
      console.log(heatVolume)
      await this.updateVolume(volume, heatVolume)
      this.working = false
    },
    async updateVolume (volume, heatVolume) {
      const realx = this.modelX = volume.xLength * volume.spacing[0]
      const realy = this.modelY = volume.yLength * volume.spacing[1]
      const realz = this.modelZ = volume.zLength * volume.spacing[2]
      const geometry = new THREE.BoxBufferGeometry(realx, realy, realz)
      geometry.translate(realx / 2 - 0.5, realy / 2 - 0.5, realz / 2 - 0.5)

      const texture = new THREE.DataTexture3D(volume.data, volume.xLength, volume.yLength, volume.zLength)
      texture.format = THREE.RedFormat
      texture.type = THREE.FloatType
      texture.minFilter = texture.magFilter = THREE.LinearFilter
      texture.unpackAlignment = 1

      const heatX = heatVolume.xLength * heatVolume.spacing[0]
      const heatY = heatVolume.yLength * heatVolume.spacing[1]
      const heatZ = heatVolume.zLength * heatVolume.spacing[2]
      const heatTexture = new THREE.DataTexture3D(heatVolume.data, heatVolume.xLength, heatVolume.yLength, heatVolume.zLength)
      heatTexture.format = THREE.RedFormat
      heatTexture.type = THREE.FloatType
      heatTexture.minFilter = heatTexture.magFilter = THREE.LinearFilter
      heatTexture.unpackAlignment = 1

      const shader = MyVolumeRenderShader
      const uniforms = THREE.UniformsUtils.clone(shader.uniforms)
      uniforms.u_data.value = texture
      uniforms.u_heat.value = heatTexture
      uniforms.u_size.value.set(realx, realy, realz)
      uniforms.u_size_h.value.set(heatX, heatY, heatZ)
      uniforms.u_min.value = this.ctwindow[0] / 100
      uniforms.u_max.value = this.ctwindow[1] / 100
      uniforms.u_clim.value.set(this.ctwindow[0] / 100, this.ctwindow[1] / 100)
      uniforms.u_clim_h.value.set(0, 1)
      uniforms.u_renderstyle.value = this.methodIdx
      uniforms.u_renderthreshold.value = this.isothres / 100
      uniforms.u_colormap.value = this.cmValues[this.cmIndex]
      uniforms.u_colormap_h.value = this.cmValues[this.heatIndex]
      uniforms.u_high.value = this.high / 100
      uniforms.u_low.value = this.low / 100
      uniforms.u_show_heat.value = this.heat
      uniforms.u_heat_rate.value = this.heatRate / 100
      uniforms.u_opacity.value = this.opacity / 100
      const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: shader.vertexShader,
        fragmentShader: shader.fragmentShader,
        side: THREE.BackSide
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(0, 0, 0)
      this.resetCamera()
      if (this.model) {
        this.scene.remove(this.model)
        this.model.geometry.dispose()
        this.model.material.dispose()
        this.texture.dispose()
        this.heatTexture.dispose()
      }
      this.scene.add(mesh)
      this.texture = texture
      this.heatTexture = heatTexture
      this.model = mesh
    },
    updateUniform () {
      if (this.model) {
        const uniforms = this.model.material.uniforms
        uniforms.u_min.value = this.ctwindow[0] / 100
        uniforms.u_max.value = this.ctwindow[1] / 100
        uniforms.u_clim.value.set(this.ctwindow[0] / 100, this.ctwindow[1] / 100)
        uniforms.u_clim_h.value.set(0, 1)
        uniforms.u_renderstyle.value = this.methodIdx
        uniforms.u_renderthreshold.value = this.isothres / 100
        uniforms.u_colormap.value = this.cmValues[this.cmIndex]
        uniforms.u_colormap_h.value = this.cmValues[this.heatIndex]
        uniforms.u_high.value = this.high / 100
        uniforms.u_low.value = this.low / 100
        uniforms.u_show_heat.value = this.heat
        uniforms.u_heat_rate.value = this.heatRate / 100
        uniforms.u_opacity.value = this.opacity / 100
      }
    },
    animate () {
      requestAnimationFrame(this.animate)
      this.renderer.render(this.scene, this.camera)
    }
  }
}
</script>

<style>

</style>
