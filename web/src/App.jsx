import { useEffect, useState } from 'react'
import { getModelInfo, getSamples, predict } from './api.js'
import { DATASET_URL, SOURCE_URL } from './config.js'
import ControlPanel from './components/ControlPanel.jsx'
import Results from './components/Results.jsx'
import Viewer from './components/Viewer.jsx'

// Turn a gallery sample or an API response into what the viewer needs.
const fromSample = (s) => ({
  id: s.id,
  cloudy: s.cloudy_url,
  cleaned: s.cleaned_url,
  clear: s.clear_url,
  changes: s.changes_url,
  psnr_input: s.psnr_input,
  psnr_model: s.psnr_model,
  changed_fraction: s.changed_fraction,
  downloadName: `cleaned_${s.id}.jpg`,
})

const fromUpload = (r, file) => ({
  id: 'upload',
  cloudy: r.input,
  cleaned: r.result,
  changes: r.changes,
  inference_ms: r.inference_ms,
  changed_fraction: r.changed_fraction,
  width: r.width,
  height: r.height,
  original_width: r.original_width,
  original_height: r.original_height,
  resized: r.resized,
  downloadName: `${file.name.replace(/\.[^.]+$/, '')}_cleaned.png`,
})

export default function App() {
  const [samples, setSamples] = useState([])
  const [info, setInfo] = useState(null)
  const [view, setView] = useState(null)
  const [mode, setMode] = useState('compare')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getSamples()
      .then((list) => {
        setSamples(list)
        if (list.length) setView(fromSample(list[0]))
      })
      .catch((e) => setError(e.message))
    getModelInfo()
      .then((i) => {
        setInfo(i)
        if (i.app_name) document.title = `${i.app_name}: remove clouds from satellite images`
      })
      .catch(() => {})
  }, [])

  const onFile = async (file) => {
    setError('')
    setBusy(true)
    try {
      const r = await predict(file)
      setView(fromUpload(r, file))
      setMode('compare')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const name = info?.app_name ?? 'Cloud Removal'

  return (
    <div className="mx-auto max-w-[76rem] px-5 sm:px-8">
      <header className="flex items-center justify-between gap-4 py-5">
        <a href="/" className="flex items-center gap-2.5 font-semibold">
          <img src="/favicon.svg" alt="" className="size-7" />
          <span className="display">{name}</span>
        </a>
        <nav className="flex gap-5 text-sm text-ink-soft">
          <a href={SOURCE_URL} className="hover:text-ink">Source code</a>
          <a href="/api/docs" className="hover:text-ink">API</a>
        </nav>
      </header>

      <main>
        <div className="grid gap-x-10 gap-y-8 pt-6 pb-16 lg:grid-cols-[minmax(0,2fr)_minmax(18rem,1fr)]">
          <div className="lg:col-span-2 max-w-[44rem]">
            <h1 className="display text-[2rem] leading-[1.1] font-bold sm:text-[2.75rem]">
              Remove clouds from satellite images
            </h1>
            <p className="mt-3 text-lg text-ink-soft">
              A Swin Transformer model fills in the ground hidden under clouds. Pick a test image or upload your own,
              then drag the handle to compare.
            </p>
          </div>
          <Viewer view={view} mode={mode} onMode={setMode} busy={busy} />
          <ControlPanel
            samples={samples}
            selectedId={view?.id}
            onSample={(s) => { setError(''); setView(fromSample(s)) }}
            onFile={onFile}
            busy={busy}
            error={error}
            view={view}
            limits={info?.limits}
          />
        </div>

        <div className="border-t border-line py-14">
          <Results info={info} />
        </div>

        <section aria-labelledby="how" className="border-t border-line py-14">
          <h2 id="how" className="display text-2xl font-semibold">How it works</h2>
          <ol className="mt-6 grid gap-8 md:grid-cols-3">
            {[
              ['Look at the image in small windows', 'The model splits the image into 8 × 8 windows and lets every part of a window compare itself with every other part (self-attention). Alternate layers shift the windows, so information spreads across the whole image.'],
              ['Zoom out, then back in', 'An encoder shrinks the image step by step to understand large shapes like cloud banks and fields. A decoder grows it back, using skip connections to recover fine detail like roads and edges.'],
              ['Correct the original', 'Instead of drawing a new image from scratch, the model predicts a correction and adds it to your image. Areas without clouds are left almost untouched.'],
            ].map(([title, text], i) => (
              <li key={title} className="max-w-prose">
                <span className="display text-sm font-semibold text-signal">Step {i + 1}</span>
                <h3 className="mt-1 font-semibold">{title}</h3>
                <p className="mt-1.5 text-ink-soft">{text}</p>
              </li>
            ))}
          </ol>

          <div className="mt-12 max-w-prose rounded-md bg-white p-5 ring-1 ring-line">
            <h3 className="font-semibold">Know the limits</h3>
            <p className="mt-1.5 text-ink-soft">
              Under thick cloud the ground is not visible at all, so the model makes an educated guess from the
              surroundings. Treat those areas as a reconstruction, not a measurement. It was trained on Google Earth
              colour images, so it works best on similar RGB imagery.
            </p>
          </div>
        </section>
      </main>

      <footer className="flex flex-wrap justify-between gap-3 border-t border-line py-8 text-sm text-ink-soft">
        <p>
          Trained on the <a className="underline underline-offset-2 hover:text-ink" href={DATASET_URL}>RICE dataset</a>{' '}
          (Lin et al., 2019).
        </p>
        <p>Built with PyTorch, FastAPI and React.</p>
      </footer>
    </div>
  )
}
