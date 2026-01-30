import { WeatherReport as WeatherReportType } from '../types';

interface Props {
  weather: WeatherReportType;
}

export function WeatherReport({ weather }: Props) {
  return (
    <div className="card weather-card">
      <div className="card-title">Market Weather</div>

      <div className="regime-indicator">
        <div className={`regime-dot ${weather.regime.risk_level}`}></div>
        <div>
          <div className="regime-name">{weather.regime.regime.replace(/_/g, ' ')}</div>
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>{weather.regime.description}</div>
        </div>
      </div>

      <div className="weather-headline">{weather.headline}</div>
      <p className="weather-summary">{weather.summary}</p>

      {weather.risks.length > 0 && (
        <>
          <div style={{ color: '#94a3b8', fontSize: '14px', marginTop: '16px', marginBottom: '8px' }}>
            Key Risks
          </div>
          <ul className="weather-risks">
            {weather.risks.map((risk, i) => (
              <li key={i}>{risk}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
