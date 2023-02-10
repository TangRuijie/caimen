module.exports = {
  transpileDependencies: [
    'vuetify'
  ],
  chainWebpack: config => {
    // GLSL Loader
    config.module
      .rule('glsl')
      .test(/\.glsl$/)
      .use('raw-loader')
      .loader('raw-loader')
      .end()
  }
}
