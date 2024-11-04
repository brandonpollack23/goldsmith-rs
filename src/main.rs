use std::time::Duration;

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use eyre::Result;
use itertools::Itertools;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    prelude::CrosstermBackend,
    widgets::{Block, Borders, Gauge},
    Terminal,
};
use rodio::{OutputStream, Sink};
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Config {
    /// Target frames per second
    #[clap(short, long, default_value_t = 30)]
    target_fps: u32,

    #[clap(short, long, default_value_t = 16)]
    num_buckets: u32,

    /// Audio file to process
    audio_file: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let config = Config::parse();

    enable_raw_mode()?;
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let (decoder, mut window_chan_rx) =
        audio::prepare_fft_decoder(&config.audio_file, config.target_fps)?;
    let visualizer = tokio::spawn(async move {
        while let Some(fft_window) = window_chan_rx.recv().await {
            let bucket_size = usize::div_ceil(fft_window.window.len(), config.num_buckets as usize);
            let mut buckets: Vec<f64> = fft_window
                .window
                .iter()
                .chunks(bucket_size)
                .into_iter()
                .map(|b| b.sum::<f64>())
                .collect();
            let max_bucket = *buckets
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            // Normalize buckets
            if max_bucket != 0.0 {
                for bucket in &mut buckets {
                    *bucket /= max_bucket;
                }
            }

            terminal
                .draw(|f| {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(
                            (0..config.num_buckets)
                                .map(|_| Constraint::Percentage(100 / config.num_buckets as u16))
                                .collect::<Vec<_>>(),
                        )
                        .split(f.area());

                    for (i, &bucket) in buckets.iter().enumerate() {
                        let gauge = Gauge::default()
                            .block(Block::default().borders(Borders::ALL))
                            .gauge_style(
                                ratatui::style::Style::default().fg(ratatui::style::Color::White),
                            )
                            .ratio(bucket);
                        f.render_widget(gauge, chunks[i]);
                    }
                })
                .expect("error drawing terminal");

            if event::poll(Duration::from_millis(0)).expect("error polling event") {
                if let Event::Key(key) = event::read().expect("error reading event") {
                    if key.code == KeyCode::Char('q')
                        || (key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c'))
                    {
                        return;
                    }
                }
            }
        }
    });

    let (_stream, stream_handle) =
        OutputStream::try_default().expect("Failed to open audio output stream");
    let sink = Sink::try_new(&stream_handle).expect("Failed to create audio sink");

    sink.append(decoder);

    visualizer.await?;

    disable_raw_mode()?;

    Ok(())
}

fn init_tracing() {
    // Console layer for tokio-console
    let console_layer = console_subscriber::ConsoleLayer::builder().spawn(); // spawns the console server in a background task

    // Fmt layer for human-readable logging
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true) // Include target (module path) in logs
        .with_level(true) // Include log levels
        .with_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(Level::INFO.into()) // Set default level to INFO
                .from_env_lossy(),
        );

    // Combine layers and set global default
    tracing_subscriber::registry()
        .with(console_layer)
        .with(fmt_layer)
        .init();
}

mod audio {
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
}
