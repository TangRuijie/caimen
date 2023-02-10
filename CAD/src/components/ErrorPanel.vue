<template>
  <v-overlay :value="active" z-index="15" opacity="0.9" @click.native="$emit('close')">
    <v-card light
      style="position: fixed; left: 200px; right: 200px; top: 200px; bottom: 200px; overflow-y: scroll;">
      <v-card-text>
        <div style="white-space:pre-wrap; font-family: Consolas, Monaco, monospace;" class="red--text">
          {{detail}}
        </div>
      </v-card-text>
    </v-card>
  </v-overlay>
</template>

<script>
export default {
  name: 'ErrorPanel',
  props: {
    detail: String,
    active: Boolean
  }
}
</script>

<style>

</style>
