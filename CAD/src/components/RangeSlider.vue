<template>
  <div>
    <span :style="`word-break: keep-all;`">{{label}}</span>
    <div class="d-flex align-center justify-start">
      <div
        @mousedown="handleMouseDown"
        @mousemove.prevent="handleMouseMove"
        @mouseup="handleMouseUp"
        @mouseleave="handleMouseLeave"
        @click="handleClick"
        :style="`height: ${height}px; width: calc(100% - ${appendWidth}); display: flex`">
        <div class="d-flex" style="width: 95%" ref="bar">
          <div :style="`width: ${getLowPercentage() * 100}%; background-color: #ECEFF1; pointer-events: none;`"></div>
          <div :style="`width: ${(getHighPercentage() - getLowPercentage()) * 100}%; background-color: #2196F3; pointer-events: none;`"></div>
          <div :style="`width: ${(1 -getHighPercentage()) * 100}%; background-color: #ECEFF1; pointer-events: none;`"></div>
        </div>
      </div>
      <span :style="`word-break: keep-all; width: ${appendWidth}`">
        {{Math.round(value[0]) / 100}}-{{Math.round(value[1]) / 100}}
      </span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RangeSlider',
  props: {
    value: Array,
    min: Number,
    max: Number,
    label: String,
    height: String,
    appendWidth: String
  },
  data: () => ({
    selecting: false
  }),
  methods: {
    getLowPercentage () {
      return (this.value[0] - this.min) / (this.max - this.min)
    },
    getHighPercentage () {
      return (this.value[1] - this.min) / (this.max - this.min)
    },
    updateValues (offsetX) {
      let l = offsetX / this.$refs.bar.clientWidth
      if (l < 0) {
        l = 0
      } else if (l > 1) {
        l = 1
      }
      const vx = this.getLowPercentage()
      const vy = this.getHighPercentage()
      const res = []
      if (Math.abs(l - vx) > Math.abs(l - vy)) {
        res.push(this.value[0])
        res.push(l * (this.max - this.min) + this.min)
      } else {
        res.push(l * (this.max - this.min) + this.min)
        res.push(this.value[1])
      }
      this.$emit('input', res)
    },
    handleMouseDown (event) {
      this.selecting = true
    },
    handleMouseMove (event) {
      if (event.buttons === 1) {
        if (this.selecting) {
          this.updateValues(event.offsetX)
        }
      } else {
        this.selecting = false
      }
    },
    handleMouseLeave (event) {
      this.selecting = false
    },
    handleMouseUp (event) {
      this.selecting = false
    },
    handleClick (event) {
      this.updateValues(event.offsetX)
    }
  }
}
</script>

<style>

</style>
