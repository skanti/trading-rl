import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parse } from 'yaml'

// config.yaml is the dashboard's single source of truth. The standalone Python daemon,
// auth provisioning script and this app read it. Only the values the browser genuinely
// needs are copied into runtimeConfig;
// the dashboard password is deliberately not among them, since Firebase Auth is what
// actually checks it.
const config = parse(
  readFileSync(fileURLToPath(new URL('./config.yaml', import.meta.url)), 'utf8')
) as {
  dashboard?: { title?: string }
  schedule?: {
    time_zone?: string
    ranking_time?: string
    entry_time?: string
    exit_time?: string
  }
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

  ui: {
    // The dashboard is small enough that native system fonts are preferable to
    // downloading dozens of Inter and JetBrains Mono variants.
    fonts: false
  },

  runtimeConfig: {
    public: {
      dashboardTitle: config.dashboard?.title ?? 'Trading dashboard',
      authEmailDomain: config.auth?.email_domain ?? '',
      firestoreCollection: config.firebase?.collection ?? 'accounts',
      firestoreDocument: config.firebase?.document ?? 'current',
      scheduleTimeZone: config.schedule?.time_zone ?? 'America/New_York',
      scheduleRankingTime: config.schedule?.ranking_time ?? '15:00',
      scheduleEntryTime: config.schedule?.entry_time ?? '15:59',
      scheduleExitTime: config.schedule?.exit_time ?? '09:00',
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
