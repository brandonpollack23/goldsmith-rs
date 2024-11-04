use eyre::Result;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use rodio::source::Buffered;
use rodio::Decoder;
use rodio::Source;
use std::time::Duration;
use std::{fs::File, io::BufReader};
use tokio::sync::mpsc::Receiver;
use tracing::debug;

#[derive(Clone, Debug)]
pub struct FFTWindow {
    pub window: Vec<f64>,
}

pub fn prepare_fft_decoder(
    audio_file_str: &str,
    target_fps: u32,
) -> Result<(Buffered<Decoder<BufReader<File>>>, Receiver<FFTWindow>)> {
    let audio_file = BufReader::new(File::open(audio_file_str)?);
    let decoder = Decoder::new(audio_file)?.buffered();

    let window_duration = Duration::from_secs(1) / target_fps;
    let fft_window_size =
        ((decoder.sample_rate() as f64) * window_duration.as_secs_f64()) as usize;

    debug!("FFT Window duration is {window_duration:?}");
    debug!("FFT sample rate is {}", decoder.sample_rate());
    debug!("FFT Window Size is {fft_window_size}");
    // let _song_duration = decoder.total_duration().unwrap();

    // fftStreamer := fft.NewFFTStreamer(ctx, streamer, fftWindowSize, format)
    let (fft_chan_tx, fft_chan_rx) = tokio::sync::mpsc::channel::<FFTWindow>(10);
    let decoder_clone = decoder.clone();
    rayon::spawn(move || {
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_window_size);

        let mut fft_window_buf: Vec<f64> = vec![];
        for sample in decoder_clone {
            fft_window_buf.push(sample as f64);
            if fft_window_buf.len() == fft_window_size {
                let mut output = vec![Complex::<f64>::new(0.0, 0.0); fft_window_size / 2 + 1];
                fft.process(&mut fft_window_buf, &mut output).unwrap();

                let window = output
                    .iter()
                    .take(output.len() / 2) // Ignore negative frequency half
                    .map(|&x| f64::sqrt(x.re * x.re + x.im * x.im))
                    .collect::<Vec<f64>>();
                if let Err(_) = fft_chan_tx.blocking_send(FFTWindow { window }) {
                    debug!("channel closed, stopping fft");
                    return;
                }

                fft_window_buf.clear();
            }
        }
    });

    Ok((decoder, fft_chan_rx))
}

