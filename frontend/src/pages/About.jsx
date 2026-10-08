export default function About() {
  return (
    <>
      <div className="page-head">
        <div className="eyebrow">About</div>
        <h1>TrafficGuard AI</h1>
        <p>An educational deep-learning project: YOLO-based analysis of Indian traffic-camera footage with a transparent, configurable rule engine.</p>
      </div>
      <div className="panel prose">
        <h3 style={{ marginTop: 0 }}>What it does</h3>
        <p>
          It analyses an uploaded image or video, detects vehicles and riders with YOLO, works out which people are sitting on which two-wheeler,
          and reports <b>AI-detected possible violations</b> — each with a confidence score, an evidence image, and a legal reference configured in a JSON file.
        </p>
        <h3>How YOLO works</h3>
        <p>
          YOLO (“You Only Look Once”) is a single-pass convolutional detector. The image is passed through a backbone network once; the network predicts, for each
          region of a grid, bounding boxes, a confidence score and class probabilities. Overlapping boxes are merged with non-maximum suppression (the IoU threshold).
          Because it is one forward pass, it is fast enough for video on a laptop. Here the small <i>nano</i> variant is used.
        </p>
        <h3>How the system reasons</h3>
        <ul>
          <li><b>Object detection</b> — YOLO says <i>what is where</i>: motorcycles, cars, buses, trucks, persons, and “person on a two-wheeler” (from the fine-tuned rider model).</li>
          <li><b>Relationship analysis</b> — geometry decides whether a person is positioned <i>on</i> a motorcycle (overlap, horizontal alignment, seating height, scale). Bystanders are not counted.</li>
          <li><b>Rule engine</b> — turns relationships into candidate violations using <span className="mono">traffic_rules.json</span>.</li>
          <li><b>Confidence &amp; validation</b> — thresholds, an “insufficient evidence” state, duplicate filtering, and (for video) temporal validation across frames.</li>
        </ul>
        <h3>Limitations</h3>
        <ul>
          <li>The training dataset has a single class (<span className="mono">person_bike</span>) and no helmet labels, so helmet compliance is <b>not assessed</b> until a helmet model is added.</li>
          <li>A single image cannot prove a red-light violation; the video red-light check relies on operator-supplied stop-line and signal timing.</li>
          <li>Heavy occlusion, night scenes, low resolution and unusual camera angles reduce accuracy. Lane violations and number-plate reading are not implemented.</li>
          <li>Small dataset (a few hundred distinct images): expect errors, and do not read metrics as real-world accuracy.</li>
        </ul>
        <h3>Educational purpose &amp; human verification</h3>
        <p>
          This is a college project, not an enforcement tool. Indian traffic law and penalties change; the legal references shipped here must be verified against current official sources.
          <b> Every result requires human verification</b> before any action is taken.
        </p>
      </div>
    </>
  );
}
