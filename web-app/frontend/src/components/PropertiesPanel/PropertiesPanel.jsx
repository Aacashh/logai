import { useState } from 'react'
import './PropertiesPanel.css'

function PropertiesPanel({ 
  wellData, 
  settings, 
  onSettingsChange,
  selectedCurve 
}) {
  // State for collapsible sections
  const [expandedSections, setExpandedSections] = useState({
    unit: true,
    dataDisplay: true,
    scale: true,
    grids: true,
    visualization: true,
    viewInterval: true,
    majorGrids: true,
    minorGrids: true
  })
  
  const toggleSection = (sectionKey) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionKey]: !prev[sectionKey]
    }))
  }
  
  const handleTrackSettingChange = (track, key, value) => {
    onSettingsChange({
      trackSettings: {
        ...settings.trackSettings,
        [track]: {
          ...settings.trackSettings[track],
          [key]: value
        }
      }
    })
  }
  
  const handleToggle = (key) => {
    onSettingsChange({ [key]: !settings[key] })
  }
  
  return (
    <aside className="properties-panel">
      {/* Header */}
      <div className="properties-header">
        <svg className="properties-header-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="6" cy="6" r="2"/>
          <path d="M8 6h6M6 8v6"/>
          <circle cx="12" cy="12" r="2"/>
          <path d="M10 12H6"/>
        </svg>
        Properties
      </div>
      
      {/* Search */}
      <div className="properties-search">
        <input type="text" placeholder="🔍 Search" />
      </div>
      
      {/* Properties Content - Scrollable */}
      <div className="properties-content">
        {/* Unit Section */}
        <div className="property-section">
          <div 
            className="property-section-header"
            onClick={() => toggleSection('unit')}
          >
            <span className={`section-arrow ${!expandedSections.unit ? 'collapsed' : ''}`}>▼</span>
            Unit
          </div>
          <div className={`property-section-content ${!expandedSections.unit ? 'collapsed' : ''}`}>
            <div className="property-grid">
              <div className="property-row">
                <span className="property-label">Null Value</span>
                <span className="property-value">-999.25</span>
              </div>
              <div className="property-row">
                <span className="property-label">Type</span>
                <span className="property-value">Well Log</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Data Display Section */}
        <div className="property-section">
          <div 
            className="property-section-header"
            onClick={() => toggleSection('dataDisplay')}
          >
            <span className={`section-arrow ${!expandedSections.dataDisplay ? 'collapsed' : ''}`}>▼</span>
            Data Display
          </div>
          <div className={`property-section-content ${!expandedSections.dataDisplay ? 'collapsed' : ''}`}>
            <div className="property-grid">
              <div className="property-row">
                <span className="property-label">Pen Color</span>
                <div className="property-value">
                  <input 
                    type="color" 
                    className="color-picker"
                    defaultValue="#CC0000"
                  />
                  <span>Red</span>
                </div>
              </div>
              <div className="property-row">
                <span className="property-label">Pen Style</span>
                <select className="property-input">
                  <option>Solid</option>
                  <option>Dashed</option>
                  <option>Dotted</option>
                </select>
              </div>
              <div className="property-row">
                <span className="property-label">Pen Width</span>
                <input 
                  type="number" 
                  className="property-input"
                  defaultValue="1.5"
                  step="0.5"
                  min="0.5"
                  max="5"
                />
              </div>
              <div className="property-row">
                <span className="property-label">Shading Position</span>
                <select className="property-input">
                  <option>None</option>
                  <option>Left</option>
                  <option>Right</option>
                </select>
              </div>
              <div className="property-row">
                <span className="property-label">Shading Type</span>
                <select className="property-input">
                  <option>Transparent</option>
                  <option>Solid</option>
                  <option>Pattern</option>
                </select>
              </div>
              <div className="property-row">
                <span className="property-label">Overflow</span>
                <select className="property-input">
                  <option>Clamp</option>
                  <option>Wrap</option>
                </select>
              </div>
              <div className="property-row">
                <span className="property-label">Filter Width</span>
                <input 
                  type="number" 
                  className="property-input"
                  value={settings.smoothing}
                  onChange={(e) => onSettingsChange({ smoothing: Number(e.target.value) })}
                  min="0"
                  max="20"
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Scale Section */}
        <div className="property-section">
          <div 
            className="property-section-header"
            onClick={() => toggleSection('scale')}
          >
            <span className={`section-arrow ${!expandedSections.scale ? 'collapsed' : ''}`}>▼</span>
            Scale
          </div>
          <div className={`property-section-content ${!expandedSections.scale ? 'collapsed' : ''}`}>
            <div className="property-grid">
              <div className="property-row">
                <span className="property-label">Low</span>
                <input 
                  type="number" 
                  className="property-input"
                  value={settings.trackSettings.gr.min}
                  onChange={(e) => handleTrackSettingChange('gr', 'min', Number(e.target.value))}
                />
              </div>
              <div className="property-row">
                <span className="property-label">High</span>
                <input 
                  type="number" 
                  className="property-input"
                  value={settings.trackSettings.gr.max}
                  onChange={(e) => handleTrackSettingChange('gr', 'max', Number(e.target.value))}
                />
              </div>
              <div className="property-row">
                <span className="property-label">Mode</span>
                <select className="property-input">
                  <option>Linear</option>
                  <option>Logarithmic</option>
                </select>
              </div>
              <div className="property-row">
                <span className="property-label">Reverse</span>
                <input type="checkbox" />
                <span>No</span>
              </div>
              <div className="property-row">
                <span className="property-label">Cleavage</span>
                <input type="checkbox" />
                <span>No</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Grids Section */}
        <div className="property-section">
          <div 
            className="property-section-header"
            onClick={() => toggleSection('grids')}
          >
            <span className={`section-arrow ${!expandedSections.grids ? 'collapsed' : ''}`}>▼</span>
            Grids
          </div>
          <div className={`property-section-content ${!expandedSections.grids ? 'collapsed' : ''}`}>
            <div className="property-grid">
              <div className="property-row">
                <span className="property-label">Overwrite Depth Grids</span>
                <input type="checkbox" defaultChecked />
                <span>Yes</span>
              </div>
            </div>
            
            {/* Major Grids */}
            <div className="property-subsection">
              <div 
                className="property-subsection-header"
                onClick={() => toggleSection('majorGrids')}
              >
                <span className={`section-arrow ${!expandedSections.majorGrids ? 'collapsed' : ''}`}>▼</span>
                Major
              </div>
              <div className={`property-section-content ${!expandedSections.majorGrids ? 'collapsed' : ''}`}>
                <div className="property-grid">
                  <div className="property-row">
                    <span className="property-label">Color</span>
                    <input type="color" className="color-picker" defaultValue="#C0C0C0" />
                    <span>Gainsboro</span>
                  </div>
                  <div className="property-row">
                    <span className="property-label">Spacing</span>
                    <input type="number" className="property-input" defaultValue="100" />
                  </div>
                  <div className="property-row">
                    <span className="property-label">Style</span>
                    <select className="property-input">
                      <option>Solid</option>
                      <option>Dashed</option>
                    </select>
                  </div>
                  <div className="property-row">
                    <span className="property-label">Width</span>
                    <input type="number" className="property-input" defaultValue="1" step="0.5" />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Minor Grids */}
            <div className="property-subsection">
              <div 
                className="property-subsection-header"
                onClick={() => toggleSection('minorGrids')}
              >
                <span className={`section-arrow ${!expandedSections.minorGrids ? 'collapsed' : ''}`}>▼</span>
                Minor
              </div>
              <div className={`property-section-content ${!expandedSections.minorGrids ? 'collapsed' : ''}`}>
                <div className="property-grid">
                  <div className="property-row">
                    <span className="property-label">Color</span>
                    <span className="property-value muted">No Color</span>
                  </div>
                  <div className="property-row">
                    <span className="property-label">Spacing</span>
                    <input type="number" className="property-input" defaultValue="2" disabled />
                  </div>
                  <div className="property-row">
                    <span className="property-label">Style</span>
                    <select className="property-input" disabled>
                      <option>Dot</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Visualization Toggles */}
        <div className="property-section">
          <div 
            className="property-section-header"
            onClick={() => toggleSection('visualization')}
          >
            <span className={`section-arrow ${!expandedSections.visualization ? 'collapsed' : ''}`}>▼</span>
            Visualization
          </div>
          <div className={`property-section-content ${!expandedSections.visualization ? 'collapsed' : ''}`}>
            <div className="property-grid">
              <div className="property-row">
                <span className="property-label">GR Sand Shading</span>
                <input 
                  type="checkbox" 
                  checked={settings.showGrFill}
                  onChange={() => handleToggle('showGrFill')}
                />
              </div>
              <div className="property-row">
                <span className="property-label">DN Crossover</span>
                <input 
                  type="checkbox" 
                  checked={settings.showDnCrossover}
                  onChange={() => handleToggle('showDnCrossover')}
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Depth Interval */}
        {wellData && (
          <div className="property-section">
            <div 
              className="property-section-header"
              onClick={() => toggleSection('viewInterval')}
            >
              <span className={`section-arrow ${!expandedSections.viewInterval ? 'collapsed' : ''}`}>▼</span>
              View Interval
            </div>
            <div className={`property-section-content ${!expandedSections.viewInterval ? 'collapsed' : ''}`}>
              <div className="property-grid">
                <div className="property-row">
                  <span className="property-label">Start</span>
                  <input 
                    type="number" 
                    className="property-input"
                    value={settings.depthRange.start?.toFixed(0) || wellData.depth_range.min}
                    step="10"
                  />
                  <span>{wellData.depth_unit}</span>
                </div>
                <div className="property-row">
                  <span className="property-label">End</span>
                  <input 
                    type="number" 
                    className="property-input"
                    value={settings.depthRange.end?.toFixed(0) || wellData.depth_range.max}
                    step="10"
                  />
                  <span>{wellData.depth_unit}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

export default PropertiesPanel
