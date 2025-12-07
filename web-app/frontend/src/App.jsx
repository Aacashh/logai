import { useState, useCallback } from 'react'
import Header from './components/Header/Header'
import Sidebar from './components/Sidebar/Sidebar'
import PropertiesPanel from './components/PropertiesPanel/PropertiesPanel'
import WellHeader from './components/WellHeader/WellHeader'
import LogViewer from './components/LogViewer/LogViewer'
import ExportPanel from './components/ExportPanel/ExportPanel'
import WelcomeScreen from './components/WelcomeScreen'
import { uploadLasFile, getWellCurves } from './services/api'
import './App.css'

function App() {
  // State
  const [wellData, setWellData] = useState(null)
  const [curveData, setCurveData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // Settings state
  const [settings, setSettings] = useState({
    scale: 500,
    smoothing: 0,
    showGrFill: true,
    showDnCrossover: true,
    depthRange: { start: null, end: null },
    trackSettings: {
      gr: { min: 0, max: 150, visible: true },
      res: { min: 0.2, max: 2000, visible: true },
      dens: { min: 1.95, max: 2.95, visible: true },
      neut: { min: -0.15, max: 0.45, visible: true },
    }
  })

  // Handle file upload
  const handleFileUpload = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await uploadLasFile(file)
      setWellData(data)
      
      // Set initial depth range
      setSettings(prev => ({
        ...prev,
        depthRange: {
          start: data.depth_range.min,
          end: Math.min(data.depth_range.min + 500, data.depth_range.max)
        }
      }))
      
      // Use curve data from upload response directly (no second API call!)
      if (data.curve_data) {
        setCurveData(data.curve_data)
      } else {
        // Fallback to fetching if not included (backwards compatibility)
        const curves = await getWellCurves(data.well_id)
        setCurveData(curves)
      }
      
    } catch (err) {
      setError(err.message || 'Failed to upload file')
      console.error('Upload error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  // Handle depth range change
  const handleDepthRangeChange = useCallback(async (start, end) => {
    if (!wellData) return
    
    setSettings(prev => ({
      ...prev,
      depthRange: { start, end }
    }))
    
    try {
      const curves = await getWellCurves(wellData.well_id, start, end)
      setCurveData(curves)
    } catch (err) {
      console.error('Error fetching curves:', err)
    }
  }, [wellData])

  // Update settings
  const handleSettingsChange = useCallback((newSettings) => {
    setSettings(prev => ({ ...prev, ...newSettings }))
  }, [])

  // Export handlers (placeholder)
  const handleExportPng = useCallback(() => {
    console.log('Export PNG - implement with html2canvas')
    // TODO: Implement with html2canvas
  }, [])

  const handleExportPdf = useCallback(() => {
    console.log('Export PDF - implement with jsPDF')
    // TODO: Implement with jsPDF
  }, [])

  return (
    <div className="app">
      <Header 
        wellData={wellData}
        settings={settings}
        onSettingsChange={handleSettingsChange}
        onExportPng={handleExportPng}
        onExportPdf={handleExportPdf}
      />
      
      <div className="app-container">
        {/* Left: Explorer Panel */}
        <Sidebar
          wellData={wellData}
          settings={settings}
          onSettingsChange={handleSettingsChange}
          onFileUpload={handleFileUpload}
          onDepthRangeChange={handleDepthRangeChange}
          loading={loading}
          curveData={curveData}
        />
        
        {/* Center: Main Content */}
        <main className="main-content">
          {error && (
            <div className="error-banner">
              <span>⚠️ {error}</span>
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}
          
          <div className="document-area">
            {/* Document Tabs */}
            {wellData && (
              <div className="document-tabs">
                <div className="document-tab active">
                  <span className="document-tab-icon">📊</span>
                  {wellData.header?.WELL || wellData.filename}
                  <button className="document-tab-close">×</button>
                </div>
              </div>
            )}
            
            {/* View Area */}
            <div className="view-area">
              {!wellData ? (
                <WelcomeScreen />
              ) : (
                <div className="view-content">
                  {/* Well Header Info */}
                  <WellHeader 
                    headerInfo={wellData.header}
                    depthUnit={wellData.depth_unit}
                  />
                  
                  {/* Log Display - Scrollable Container */}
                  <div className="log-viewport">
                    <LogViewer
                      wellData={wellData}
                      curveData={curveData}
                      settings={settings}
                      loading={loading}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
        
        {/* Right: Properties Panel */}
        <PropertiesPanel
          wellData={wellData}
          settings={settings}
          onSettingsChange={handleSettingsChange}
        />
      </div>
      
      {/* Status Bar */}
      <div className="status-bar">
        <span className="status-bar-section">
          {wellData ? `Well: ${wellData.header?.WELL || 'Unknown'}` : 'No file loaded'}
        </span>
        <span className="status-bar-section">
          Scale: 1:{settings.scale}
        </span>
        {wellData && (
          <span className="status-bar-section">
            Depth: {settings.depthRange.start?.toFixed(0)} - {settings.depthRange.end?.toFixed(0)} {wellData.depth_unit}
          </span>
        )}
        <span style={{ flex: 1 }}></span>
        <span className="status-bar-section">
          WellLog Viewer v1.0
        </span>
      </div>
    </div>
  )
}

export default App
