import { useRef, useState } from 'react'

/** Right-hand column: upload, test-image gallery, and facts about the current image. */
export default function ControlPanel({ samples, selectedId, onSample, onFile, busy, error, view, limits }) {
  const input = useRef(null)
  const [over, setOver] = useState(false)
  const maxMb = limits?.max_upload_mb ?? 10
  const maxSide = limits?.max_side ?? 1024

  const pick = (files) => {
    const f = files?.[0]
    if (f) onFile(f)
  }

  return (
    <div className="flex flex-col gap-7">
      <div>
        <h2 className="display text-lg font-semibold">Use your own image</h2>
        <label
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) pick(e.dataTransfer.files) }}
          className={`mt-3 flex cursor-pointer flex-col items-start gap-1 rounded-md border-2 border-dashed p-4 transition-colors
            ${over ? 'border-signal bg-white' : 'border-line bg-white/60 hover:bg-white'}
            ${busy ? 'pointer-events-none opacity-60' : ''}`}
        >
          <span className="font-medium">Drop a satellite image here</span>
          <span className="text-sm text-ink-soft">
            or <span className="font-medium text-signal underline underline-offset-2">choose a file</span>.
            {' '}PNG, JPEG, TIFF or WebP, up to {maxMb} MB. Images wider than {maxSide} px are scaled down.
          </span>
          <input
            ref={input}
            type="file"
            accept="image/png,image/jpeg,image/tiff,image/webp,.tif,.tiff"
            className="sr-only"
            disabled={busy}
            onChange={(e) => { pick(e.target.files); e.target.value = '' }}
          />
        </label>
        {error && (
          <p role="alert" className="mt-2.5 rounded-sm bg-signal/10 px-3 py-2 text-sm text-signal-dark">{error}</p>
        )}
      </div>

      {samples.length > 0 && (
        <div>
          <h2 className="display text-lg font-semibold">Or try a test image</h2>
          <p className="mt-1 text-sm text-ink-soft">
            From the RICE1 test set, which the model never saw while training. Picked across the score range, best to worst.
          </p>
          <ul className="mt-3 grid grid-cols-3 gap-2">
            {samples.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => onSample(s)}
                  disabled={busy}
                  aria-pressed={selectedId === s.id}
                  aria-label={`Test image ${s.id}`}
                  className={`block w-full overflow-hidden rounded-sm ring-offset-2 ring-offset-mineral transition
                    ${selectedId === s.id ? 'ring-2 ring-signal' : 'ring-1 ring-line hover:ring-ink-soft'}`}
                >
                  <img src={s.cloudy_url} alt="" loading="lazy" className="aspect-square w-full object-cover" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {view && <Facts view={view} />}
    </div>
  )
}

function Facts({ view }) {
  const rows = []
  if (view.psnr_model != null) {
    rows.push(['Match with the real image (PSNR)', `${view.psnr_input.toFixed(1)} dB → ${view.psnr_model.toFixed(1)} dB`])
  }
  if (view.inference_ms != null) rows.push(['Processing time', `${(view.inference_ms / 1000).toFixed(1)} s on a 2-core CPU`])
  if (view.changed_fraction != null) rows.push(['Pixels changed noticeably', `${Math.round(view.changed_fraction * 100)}%`])
  if (view.width) rows.push(['Size', `${view.width} × ${view.height} px${view.resized ? ` (scaled down from ${view.original_width} × ${view.original_height})` : ''}`])

  return (
    <div className="border-t border-line pt-5">
      <dl className="grid gap-2.5 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4">
            <dt className="text-ink-soft">{k}</dt>
            <dd className="text-right font-medium">{v}</dd>
          </div>
        ))}
      </dl>
      <a
        href={view.cleaned}
        download={view.downloadName}
        className="mt-5 inline-flex rounded-full bg-signal px-4 py-2 text-sm font-semibold text-white hover:bg-signal-dark"
      >
        Download cleaned image
      </a>
    </div>
  )
}
