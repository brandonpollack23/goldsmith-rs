use std::{io::Stdout, time::Duration};

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
    style::Color,
    widgets::{Block, Borders, Gauge},
    Terminal,
};
use rodio::{OutputStream, Sink};
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

mod audio;

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

    let (decoder, window_chan_rx) =
        audio::prepare_fft_decoder(&config.audio_file, config.target_fps)?;
    let visualizer = tokio::spawn(async move {
        visualizer_loop(&config, &mut terminal, window_chan_rx).await;
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

async fn visualizer_loop(
    config: &Config,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut window_chan_rx: tokio::sync::mpsc::Receiver<audio::FFTWindow>,
) {
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
                        .gauge_style(ratatui::style::Style::default().fg(color_gradient(bucket)))
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
}

fn color_gradient(value: f64) -> Color {
    let r = (255.0 * value) as u8;
    let g = (255.0 * (1.0 - value)) as u8;
    Color::Rgb(r, g, 0)
}
