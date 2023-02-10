<template>
  <div>
    <span :style="`word-break: keep-all;`">{{label}}</span>
    <div class="d-flex align-center justify-start">
      <div
        @mousedown="handleMouseDown"
        @mousemove.prevent="handleMouseMove"
        @mouseup="handleMouseUp"
        @click="handleClick"
        :style="`height: ${height}px; width: calc(100% - ${appendWidth}); display: flex`">
        <div class="d-flex" style="width: 95%" ref="bar">
          <div :style="`width: ${(getPercentage()) * 100}%; background-color: #2196F3; pointer-events: none;`"></div>
          <div :style="`width: ${(1 -getPercentage()) * 100}%; background-color: #ECEFF1; pointer-events: none;`"></div>
        </div>
      </div>
      <span v-if="appendType==='value'"
        :style="`word-break: keep-all; width: ${appendWidth}`"
      ><input ref="input" v-model="buffer" type="number" :style="`width: ${appendWidth}`" @blur="finishInput" @keypress.enter="handleEnter">
      </span>
      <span v-else :style="`word-break: keep-all; width: ${appendWidth}`">{{Math.round(value) / 100}}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RangeSlider',
  props: {
    value: Number,
    min: Number,
    max: Number,
    label: String,
    height: String,
    appendWidth: String,
    appendType: String
  },
  data: () => ({
    selecting: false,
    buffer: 0
  }),
  watch: {
    value: function (newval, oldval) {
      this.buffer = newval
    }
  },
  mounted () {
    this.buffer = this.value
  },
  methods: {
    getPercentage () {
      return (this.value - this.min) / (this.max - this.min)
    },
    updateValue (offsetX) {
      let l = offsetX / this.$refs.bar.clientWidth
      if (l < 0) {
        l = 0
      } else if (l > 1) {
        l = 1
      }
      const res = l * (this.max - this.min) + this.min
      this.$emit('input', Math.round(res))
    },
    handleMouseDown (event) {
      this.selecting = true
    },
    handleMouseMove (event) {
      if (event.buttons === 1) {
        if (this.selecting) {
          this.updateValue(event.offsetX)
        }
      } else {
        this.selecting = false
      }
    },
    handleMouseUp (event) {
      this.selecting = false
    },
    handleClick (event) {
      this.updateValue(event.offsetX)
    },
    finishInput (event) {
      const num = parseInt(this.buffer)
      if (isNaN(num)) {
        this.buffer = this.value
      } else {
        this.$emit('input', num)
      }
    },
    handleEnter (event) {
      this.$refs.input.blur()
    }
  }
}
</script>

<style>

</style>
