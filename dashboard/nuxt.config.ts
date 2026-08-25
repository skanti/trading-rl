import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parse } from 'yaml'

// config.yaml is the single source of truth for the whole project -- the Python
// publisher, the digest email and this app all read it, so there is no .env to keep
// in sync. Only the values the browser genuinely needs are copied into runtimeConfig;
// the dashboard password is deliberately not among them, since Firebase Auth is what
// actually checks it.
const config = parse(
  readFileSync(fileURLToPath(new URL('./config.yaml', import.meta.url)), 'utf8')
) as {
  dashboard?: { title?: string }
  auth?: { email_domain?: string }
  firebase?: {
    collection?: string
    document?: string
    web?: Record<string, string>
  }
}

const web = config.firebase?.web ?? {}

export default defineNuxtConfig({
  modules: [
    '@nuxt/eslint',
    '@nuxt/ui'
  ],

  // SPA only -- the app is served as static files from Firebase Hosting, and every
  // credentialled call happens server-side in the Python publisher instead.
  ssr: false,

  devtools: {
    enabled: true
  },

  app: {
    head: {
      htmlAttrs: {
        lang: 'en'
      },
      title: config.dashboard?.title ?? 'Trading dashboard',
      meta: [
        { name: 'description', content: 'Live performance of the overnight-liquidity trading account.' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' },
        { name: 'theme-color', content: '#020617' },
        { name: 'robots', content: 'noindex, nofollow' }
      ],
      link: [
        { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }
      ]
    }
  },

  css: ['~/assets/css/main.css'],

  colorMode: {
    preference: 'dark'
  },

  runtimeConfig: {
    public: {
      dashboardTitle: config.dashboard?.title ?? 'Trading dashboard',
      authEmailDomain: config.auth?.email_domain ?? '',
      firestoreCollection: config.firebase?.collection ?? 'accounts',
      firestoreDocument: config.firebase?.document ?? 'paper',
      // Nuxt applies NUXT_PUBLIC_* overrides on top of these automatically, so setting
      // NUXT_PUBLIC_FIREBASE_API_KEY= (empty) forces the offline demo adapter.
      firebaseApiKey: web.apiKey ?? '',
      firebaseAuthDomain: web.authDomain ?? '',
      firebaseProjectId: web.projectId ?? '',
      firebaseStorageBucket: web.storageBucket ?? '',
      firebaseMessagingSenderId: web.messagingSenderId ?? '',
      firebaseAppId: web.appId ?? ''
    }
  },

  devServer: {
    port: 5001
  },

  compatibilityDate: '2025-01-15',

  eslint: {
    config: {
      stylistic: {
        commaDangle: 'never',
        braceStyle: '1tbs'
      }
    }
  }
})
