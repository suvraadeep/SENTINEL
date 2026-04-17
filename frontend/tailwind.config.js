/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sentinel: {
          bg:        '#0A0E1A',
          surface:   '#111827',
          card:      '#1A2235',
          border:    '#1E293B',
          hover:     '#243048',
          blue:      '#3B82F6',
          cyan:      '#06B6D4',
          purple:    '#8B5CF6',
          green:     '#10B981',
          yellow:    '#F59E0B',
          red:       '#EF4444',
          pink:      '#EC4899',
          teal:      '#14B8A6',
          text:      '#F1F5F9',
          muted:     '#94A3B8',
          faint:     '#475569',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'sentinel-grid': `linear-gradient(rgba(59, 130, 246, 0.03) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(59, 130, 246, 0.03) 1px, transparent 1px)`,
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      animation: {
        'pulse-slow':   'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in':     'slideIn 0.3s ease-out',
        'fade-in':      'fadeIn 0.4s ease-out',
        'scan':         'scan 3s linear infinite',
        'glow':         'glow 2s ease-in-out infinite alternate',
        'shimmer':      'shimmer 2s linear infinite',
      },
      keyframes: {
        slideIn: {
          '0%':   { transform: 'translateY(10px)', opacity: 0 },
          '100%': { transform: 'translateY(0)',    opacity: 1 },
        },
        fadeIn: {
          '0%':   { opacity: 0 },
          '100%': { opacity: 1 },
        },
        scan: {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        glow: {
          from: { boxShadow: '0 0 10px rgba(59,130,246,0.3)' },
          to:   { boxShadow: '0 0 20px rgba(6,182,212,0.5)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      boxShadow: {
        'sentinel': '0 0 0 1px rgba(59,130,246,0.2), 0 4px 16px rgba(0,0,0,0.4)',
        'glow-blue': '0 0 20px rgba(59,130,246,0.4)',
        'glow-cyan': '0 0 20px rgba(6,182,212,0.4)',
      },
    },
  },
  plugins: [],
}
