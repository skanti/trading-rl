export default defineAppConfig({
  ui: {
    colors: {
      primary: 'emerald',
      neutral: 'slate'
    },
    card: {
      slots: {
        header: 'p-2 sm:p-2',
        body: 'p-2 sm:p-2',
        footer: 'p-2 sm:p-2'
      }
    },
    table: {
      slots: {
        th: 'px-2 py-2 text-xs',
        td: 'px-2 py-2 text-sm'
      }
    }
  }
})
