'use strict'

import { app, protocol, BrowserWindow, ipcMain, dialog } from 'electron'
import { createProtocol } from 'vue-cli-plugin-electron-builder/lib'
import fs from 'fs'
import { spawn } from 'child_process'
// import installExtension, { VUEJS_DEVTOOLS } from 'electron-devtools-installer'
const isDevelopment = process.env.NODE_ENV !== 'production'

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let win

// Scheme must be registered before the app is ready
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { secure: true, standard: true } }
])

const openFile = function (path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, (err, data) => {
      if (err) {
        reject(err)
      } else {
        resolve(data)
      }
    })
  })
}

const execProc = function (cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { cwd: cwd })
    proc.once('close', code => {
      if (code === 0) {
        resolve(code)
      } else {
        reject(code)
      }
    })
    proc.stdout.on('data', data => {
      if (win) {
        win.webContents.send('stdOut', data)
      }
    })
    proc.stderr.on('data', data => {
      if (win) {
        win.webContents.send('stdErr', data)
      }
    })
  })
}

function createWindow () {
  // Create the browser window.
  win = new BrowserWindow({
    minWidth: 1280,
    minHeight: 720,
    width: 1280,
    height: 720,
    frame: process.env.NODE_ENV !== 'production',
    webPreferences: {
      // Use pluginOptions.nodeIntegration, leave this alone
      // See nklayman.github.io/vue-cli-plugin-electron-builder/guide/security.html#node-integration for more info
      nodeIntegration: true,
      enableRemoteModule: true
    }
  })

  win.on('maximize', e => {
    win.webContents.send('maximize')
  })

  win.on('unmaximize', e => {
    win.webContents.send('unmaximize')
  })

  win.on('restore', e => {
    win.webContents.send('restore')
  })

  ipcMain.handle('closeWindow', (event) => {
    win.close()
  })

  ipcMain.handle('maxWindow', (event) => {
    if (win.isMaximized()) {
      win.restore()
    } else {
      win.maximize()
    }
  })

  ipcMain.handle('minWindow', (event) => {
    win.minimize()
  })

  ipcMain.handle('openDir', event => {
    return dialog.showOpenDialog(win, {
      properties: ['openDirectory']
    })
  })

  ipcMain.handle('selectFile', (event, filters) => {
    return dialog.showOpenDialog(win, {
      filters: filters
    })
  })

  ipcMain.handle('openFile', (event, path) => {
    return openFile(path)
  })

  ipcMain.handle('execProc', (event, cmd, args, cwd) => {
    return execProc(cmd, args, cwd)
  })

  if (process.env.WEBPACK_DEV_SERVER_URL) {
    // Load the url of the dev server if in development mode
    win.loadURL(process.env.WEBPACK_DEV_SERVER_URL)
    if (!process.env.IS_TEST) win.webContents.openDevTools()
  } else {
    createProtocol('app')
    // Load the index.html when not in development
    win.loadURL('app://./index.html')
  }

  win.on('closed', () => {
    win = null
  })
}

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (win === null) {
    createWindow()
  }
})

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', async () => {
  if (isDevelopment && !process.env.IS_TEST) {
    // Install Vue Devtools
    try {
      // await installExtension(VUEJS_DEVTOOLS)
    } catch (e) {
      console.error('Vue Devtools failed to install:', e.toString())
    }
  }
  createWindow()
})

// Exit cleanly on request from parent process in development mode.
if (isDevelopment) {
  if (process.platform === 'win32') {
    process.on('message', (data) => {
      if (data === 'graceful-exit') {
        app.quit()
      }
    })
  } else {
    process.on('SIGTERM', () => {
      app.quit()
    })
  }
}
