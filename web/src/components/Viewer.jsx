import CompareSlider from './CompareSlider.jsx'

const MODES = [
  { id: 'compare', label: 'Before and after' },
  { id: 'changes', label: 'What changed' },
  { id: 'truth', label: 'Against the real image', samplesOnly: true },
]

/** The big image area: mode switch + slider / change map, with a working state. */
export default function Viewer({ view, mode, onMode, busy }) {
  const modes = MODES.filter((m) => !m.samplesOnly || view?.clear)
  const active = modes.some((m) => m.id === mode) ? mode : 'compare'

  return (
    <section aria-label="Result viewer">
      <div role="group" aria-label="View" className="mb-3 flex flex-wrap gap-1.5">
        {modes.map((m) => (
          <button
            key={m.id}
            type="button"
            aria-pressed={active === m.id}
            disabled={!view}
            onClick={() => onMode(m.id)}
            className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors
              ${active === m.id ? 'bg-ink text-white' : 'bg-white text-ink-soft hover:text-ink ring-1 ring-line'}`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="relative overflow-hidden rounded-md bg-mineral-deep ring-1 ring-line">
        {!view && <div className="aspect-square animate-pulse" />}

        {view && active === 'compare' && (
          <CompareSlider before={view.cloudy} after={view.cleaned} beforeLabel="Cloudy" afterLabel="Cleaned" />
        )}
        {view && active === 'truth' && (
          <CompareSlider before={view.cleaned} after={view.clear} beforeLabel="Cleaned" afterLabel="Real clear image" />
        )}
        {view && active === 'changes' && (
          <div className="relative">
            <img src={view.cloudy} alt="Cloudy input" className="block w-full h-auto" />
            <img src={view.changes} alt="Areas the model changed, in amber" className="absolute inset-0 w-full h-full" />
          </div>
        )}

        {busy && (
          <div className="absolute inset-0 grid place-items-center bg-ink/55" role="status">
            <div className="scanline absolute inset-x-0 top-0 h-[10%] bg-gradient-to-b from-transparent via-white/25 to-transparent" />
            <p className="relative rounded-sm bg-ink px-3 py-1.5 text-sm text-white">Removing clouds…</p>
          </div>
        )}
      </div>

      <p className="mt-2.5 min-h-[1.5em] text-sm text-ink-soft">
        {view && active === 'compare' && 'Drag the handle, or focus it and use the arrow keys.'}
        {view && active === 'changes' && (
          <><span className="mr-1.5 inline-block size-2.5 rounded-full bg-change align-middle" />Amber marks where the model changed the image. Stronger amber means a bigger change.</>
        )}
        {view && active === 'truth' && 'The real image was taken without clouds. The closer the two halves match, the better the model did.'}
      </p>
    </section>
  )
}
