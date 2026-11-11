/** Test-set numbers, read from results/metrics.json via /api/model. */
export default function Results({ info }) {
  const m = info?.metrics

  return (
    <section aria-labelledby="results" className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_1.4fr] [&>*]:min-w-0">
      <div className="max-w-prose">
        <h2 id="results" className="display text-2xl font-semibold">How well it works</h2>
        {m ? (
          <p className="mt-3 text-ink-soft">
            Measured on {m.test_images} {m.dataset} images that were held out from training, at their full{' '}
            {m.image_size[0]} × {m.image_size[1]} size. The first row scores the cloudy image itself, so you can
            see how much the model adds. On average it improves the match by{' '}
            <strong className="font-semibold text-ink">{m.psnr_gain_db.mean.toFixed(1)} dB</strong>, and it improves{' '}
            {m.images_improved_pct}% of test images.
          </p>
        ) : (
          <p className="mt-3 text-ink-soft">Test results will appear here once the model has been evaluated.</p>
        )}
      </div>

      {m && (
        <div>
          <div className="overflow-x-auto rounded-md bg-white ring-1 ring-line">
            <table className="w-full min-w-[30rem] text-left text-sm">
              <thead className="border-b border-line text-ink-soft">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Method</th>
                  <th scope="col" className="px-4 py-3 font-medium">PSNR</th>
                  <th scope="col" className="px-4 py-3 font-medium">SSIM</th>
                  <th scope="col" className="px-4 py-3 font-medium">Correlation</th>
                </tr>
              </thead>
              <tbody>
                <Row name="Cloudy image, no processing" s={m.no_processing} />
                <Row name="This model" s={m.model} strong />
              </tbody>
            </table>
          </div>
          <dl className="mt-4 grid gap-3 text-sm text-ink-soft sm:grid-cols-3">
            <div><dt className="font-medium text-ink">PSNR, in dB</dt><dd>How close the pixel values are. Higher is better; 1 dB more is clearly visible.</dd></div>
            <div><dt className="font-medium text-ink">SSIM, 0 to 1</dt><dd>How alike the edges and textures look. 1 is identical.</dd></div>
            <div><dt className="font-medium text-ink">Correlation, −1 to 1</dt><dd>Whether bright and dark areas line up with the real image.</dd></div>
          </dl>
          <p className="mt-4 text-sm text-ink-soft">
            {m.parameters.toLocaleString()} parameters
            {m.ms_per_image != null && <>, {Math.round(m.ms_per_image)} ms per image on {m.device}</>}.
          </p>
        </div>
      )}
    </section>
  )
}

function Row({ name, s, strong }) {
  const cell = (k, d) => (
    <td className="whitespace-nowrap px-4 py-3">
      {s[k].mean.toFixed(d)}
      <span className="ml-1 text-xs text-ink-soft">± {s[k].std.toFixed(d)}</span>
    </td>
  )
  return (
    <tr className={strong ? 'font-semibold' : 'border-b border-line'}>
      <th scope="row" className="px-4 py-3 font-[inherit]">{name}</th>
      {cell('psnr', 2)}
      {cell('ssim', 3)}
      {cell('cc', 3)}
    </tr>
  )
}
