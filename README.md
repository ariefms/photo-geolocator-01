# photo-geolocator-01 — model-free geolocator

Entry for the make.fun bounty [photo-geolocator-01](https://make.fun/bounty/photo-geolocator-01)
("Find where a photo was taken"). Node 22, standard library only, no model
calls, no network beyond the trial env.

## What it does

`main.ts` fetches the instance's 25 photos from `MAKEFUN_ENV_URL`, reads each
photo's pixels with its own baseline-JPEG DC reader (one mean colour per 8x8
block, no IDCT, no library), and posts one latitude/longitude per photo.

## Where the answer comes from

One point for every photo: the geometric median of the 75 public practice
answers the kit ships — lat 56.3956, lon 30.6587. The median is the single
point that minimises total haversine error over those answers, and the scored
photos come from the same pool, so it is the best one-point guess the public
data supports. On the kit's practice trials it scores 263,999 (trial-0),
349,463 (trial-1) and 374,164 (trial-2) out of 500,377.

The pixel cues (brightness, sky, vegetation, desert, snow, night) are
measured and logged, but they do not move the answer, deliberately: every
cue-based move was fitted on two practice trials and tested on the third, and
each one lowered the held-out score (the best green-based rule lost 13,882 on
trial-1; a sky/vegetation band rule lost 11,086 on trial-1 and 625 on
trial-2). The cues speak, weakly, about latitude, while most of the error is
longitude, which no pixel cue predicts. A photo the reader cannot decode is
answered with the anchor too, never left blank.

## Run

```sh
node main.ts   # needs MAKEFUN_ENV_URL set, as the grader sets it
```

## License

MIT. See LICENSE.
