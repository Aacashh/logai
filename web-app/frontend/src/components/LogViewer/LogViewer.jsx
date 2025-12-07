import { useRef, useEffect, useMemo } from 'react'
import * as d3 from 'd3'
import './LogViewer.css'

function LogViewer({ wellData, curveData, settings, loading }) {
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  
  // Track configuration - WellCAD style
  const trackConfig = useMemo(() => ({
    depth: { 
      width: 60, // Slightly wider for readability
      title: 'Depth',
      unit: wellData?.depth_unit || 'm'
    },
    gr: { 
      width: 250, 
      title: 'Gamma Ray',
      color: '#00AA00', // Green
      unit: 'gAPI',
      min: settings.trackSettings.gr.min,
      max: settings.trackSettings.gr.max,
      scale: 'linear'
    },
    res: { 
      width: 250, 
      title: 'Resistivity',
      subtitle: 'ohm.m',
      color: '#0000FF', // Deep Blue
      medColor: '#FF0000', // Med Red
      shalColor: '#FFA500', // Shal Orange
      unit: 'ohm.m',
      min: settings.trackSettings.res.min,
      max: settings.trackSettings.res.max,
      scale: 'log'
    },
    dn: { 
      width: 250, 
      title: 'Porosity & Density',
      subtitle: 'DN',
      densColor: '#FF0000', // Red
      neutColor: '#0000FF', // Blue
      densUnit: 'g/cc',
      neutUnit: 'v/v',
      densMin: settings.trackSettings.dens.min,
      densMax: settings.trackSettings.dens.max,
      neutMin: settings.trackSettings.neut.min,
      neutMax: settings.trackSettings.neut.max,
    }
  }), [settings.trackSettings, wellData?.depth_unit])
  
  // Calculate dimensions
  const dimensions = useMemo(() => {
    const headerHeight = 40
    const totalWidth = trackConfig.depth.width + trackConfig.gr.width + 
                       trackConfig.res.width + trackConfig.dn.width + 20
    
    // Calculate height based on depth range and scale
    // At 1:500 scale, 1 meter = 0.2 cm on plot, so 500m = 100cm = 1000px (approx)
    const depthRange = (settings.depthRange.end || 0) - (settings.depthRange.start || 0)
    // Convert to plot height: depth * (100 cm per unit at 1:1) / scale * pixels per cm
    // For better readability, use 50 pixels per cm (more zoomed in than typical 37.8)
    const heightCm = (depthRange * 100) / settings.scale
    const pxPerCm = 50 // Increased for better visibility
    const heightPx = Math.max(800, heightCm * pxPerCm) // Min 800px, no max cap for scrolling
    
    return {
      width: totalWidth,
      height: heightPx + headerHeight,
      headerHeight,
      plotHeight: heightPx
    }
  }, [settings, trackConfig])
  
  // Render chart with D3
  useEffect(() => {
    if (!curveData || !svgRef.current) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    
    const { width, height, headerHeight, plotHeight } = dimensions
    
    // Depth data
    const depth = curveData.depth || []
    const curves = curveData.curves || {}
    const mapping = curveData.mapping || {}
    
    if (depth.length === 0) return
    
    // Depth scale
    const depthMin = d3.min(depth)
    const depthMax = d3.max(depth)
    const yScale = d3.scaleLinear()
      .domain([depthMin, depthMax])
      .range([headerHeight, headerHeight + plotHeight])
    
    // Optimized line generator with curve simplification for better rendering performance
    const createLine = (xScale) => {
      return d3.line()
        .defined(d => d.value !== null && !isNaN(d.value))
        .x(d => xScale(d.value))
        .y(d => yScale(d.depth))
        .curve(d3.curveMonotoneY) // Smoother curves with less overhead
    }
    
    // Draw track header box - WellCAD style
    const drawTrackHeader = (group, config, x, w) => {
      // Header background
      group.append('rect')
        .attr('x', x)
        .attr('y', 0)
        .attr('width', w)
        .attr('height', headerHeight)
        .attr('fill', '#F0F0F0')
        .attr('stroke', '#808080')
        .attr('stroke-width', 1)
      
      // Title box
      group.append('rect')
        .attr('x', x + 2)
        .attr('y', 2)
        .attr('width', w - 4)
        .attr('height', 18)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      // Title text
      group.append('text')
        .attr('x', x + w / 2)
        .attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .text(config.title)
      
      // Scale range box
      group.append('rect')
        .attr('x', x + 2)
        .attr('y', 22)
        .attr('width', w - 4)
        .attr('height', 14)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      // Scale range text
      if (config.min !== undefined) {
        group.append('text')
          .attr('x', x + w / 2)
          .attr('y', 33)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('fill', config.color || '#333')
          .text(`${config.min} ${config.unit || ''} ${config.max}`)
      }
    }
    
    // Draw track background with grid
    const drawTrackBackground = (group, x, w, startY, h) => {
      // White background
      group.append('rect')
        .attr('x', x)
        .attr('y', startY)
        .attr('width', w)
        .attr('height', h)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#808080')
      
      // Horizontal grid lines (depth)
      const depthStep = 5 // meters
      for (let d = Math.ceil(depthMin / depthStep) * depthStep; d <= depthMax; d += depthStep) {
        const y = yScale(d)
        const isMajor = d % 10 === 0
        group.append('line')
          .attr('x1', x)
          .attr('x2', x + w)
          .attr('y1', y)
          .attr('y2', y)
          .attr('stroke', isMajor ? '#C0C0C0' : '#E8E8E8')
          .attr('stroke-width', isMajor ? 1 : 0.5)
      }
    }
    
    let xOffset = 0
    
    // ========================================
    // DEPTH TRACK
    // ========================================
    const depthGroup = svg.append('g').attr('class', 'track track-depth')
    
    // Header
    depthGroup.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', trackConfig.depth.width)
      .attr('height', headerHeight)
      .attr('fill', '#F0F0F0')
      .attr('stroke', '#808080')
    
    depthGroup.append('text')
      .attr('x', trackConfig.depth.width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text('Depth')
    
    depthGroup.append('text')
      .attr('x', trackConfig.depth.width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .text(`1m:${settings.scale / 100}m`)
    
    // Depth axis background
    depthGroup.append('rect')
      .attr('x', 0)
      .attr('y', headerHeight)
      .attr('width', trackConfig.depth.width)
      .attr('height', plotHeight)
      .attr('fill', '#FAFAFA')
      .attr('stroke', '#808080')
    
    // Depth labels
    const depthStep = 5
    for (let d = Math.ceil(depthMin / depthStep) * depthStep; d <= depthMax; d += depthStep) {
      const y = yScale(d)
      const isMajor = d % 10 === 0
      
      if (isMajor) {
        depthGroup.append('text')
          .attr('x', trackConfig.depth.width - 4)
          .attr('y', y + 3)
          .attr('text-anchor', 'end')
          .attr('font-size', '9px')
          .attr('fill', '#333')
          .text(d.toFixed(1))
      }
      
      depthGroup.append('line')
        .attr('x1', trackConfig.depth.width - (isMajor ? 8 : 4))
        .attr('x2', trackConfig.depth.width)
        .attr('y1', y)
        .attr('y2', y)
        .attr('stroke', '#666')
    }
    
    xOffset = trackConfig.depth.width
    
    // ========================================
    // GAMMA RAY TRACK  
    // ========================================
    const grKey = mapping.GR
    if (grKey && curves[grKey]) {
      const grGroup = svg.append('g')
        .attr('class', 'track track-gr')
      
      drawTrackHeader(grGroup, trackConfig.gr, xOffset, trackConfig.gr.width)
      drawTrackBackground(grGroup, xOffset, trackConfig.gr.width, headerHeight, plotHeight)
      
      // Scale
      const grScale = d3.scaleLinear()
        .domain([trackConfig.gr.min, trackConfig.gr.max])
        .range([xOffset, xOffset + trackConfig.gr.width])
      
      // Vertical grid
      grScale.ticks(5).forEach(tick => {
        svg.append('line')
          .attr('x1', grScale(tick))
          .attr('x2', grScale(tick))
          .attr('y1', headerHeight)
          .attr('y2', headerHeight + plotHeight)
          .attr('stroke', '#E0E0E0')
      })
      
      // Data
      const grData = depth.map((d, i) => ({
        depth: d,
        value: curves[grKey][i]
      }))
      
      // GR fill if enabled
      if (settings.showGrFill) {
        const area = d3.area()
          .defined(d => d.value !== null && !isNaN(d.value))
          .x0(d => grScale(d.value))
          .x1(xOffset + trackConfig.gr.width) // Fill to Right Edge
          .y(d => yScale(d.depth))
        
        grGroup.append('path')
          .datum(grData)
          .attr('d', area)
          .attr('fill', '#FFFF00') // Yellow for Sand
          .attr('fill-opacity', 0.5)
      }
      
      // Line
      const grLine = createLine(grScale)
      grGroup.append('path')
        .datum(grData)
        .attr('d', grLine)
        .attr('fill', 'none')
        .attr('stroke', trackConfig.gr.color)
        .attr('stroke-width', 1.5)
      
      xOffset += trackConfig.gr.width
    }
    
    // ========================================
    // RESISTIVITY TRACK
    // ========================================
    const resDeepKey = mapping.RES_DEEP
    const resMedKey = mapping.RES_MED
    const resShalKey = mapping.RES_SHAL

    if ((resDeepKey && curves[resDeepKey]) || (resMedKey && curves[resMedKey]) || (resShalKey && curves[resShalKey])) {
      const resGroup = svg.append('g')
        .attr('class', 'track track-res')
      
      // Modified header for resistivity
      const resConfig = {...trackConfig.res}
      drawTrackHeader(resGroup, resConfig, xOffset, trackConfig.res.width)
      drawTrackBackground(resGroup, xOffset, trackConfig.res.width, headerHeight, plotHeight)
      
      // Log scale
      const resScale = d3.scaleLog()
        .domain([Math.max(0.1, trackConfig.res.min), trackConfig.res.max])
        .range([xOffset, xOffset + trackConfig.res.width])
        .clamp(true)
      
      // Log grid
      ;[0.2, 1, 10, 100, 1000].forEach(tick => {
        if (tick >= trackConfig.res.min && tick <= trackConfig.res.max) {
          svg.append('line')
            .attr('x1', resScale(tick))
            .attr('x2', resScale(tick))
            .attr('y1', headerHeight)
            .attr('y2', headerHeight + plotHeight)
            .attr('stroke', tick === 1 || tick === 10 || tick === 100 ? '#C0C0C0' : '#E8E8E8')
            .attr('stroke-width', tick === 1 || tick === 10 || tick === 100 ? 1 : 0.5)
        }
      })
      
      // Helper to draw res curve
      const drawResCurve = (key, color) => {
        if (key && curves[key]) {
           const data = depth.map((d, i) => ({
            depth: d,
            value: curves[key][i]
          }))
          const line = createLine(resScale)
          resGroup.append('path')
            .datum(data)
            .attr('d', line)
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 1.5)
        }
      }

      // Draw in order: Deep -> Med -> Shallow (so shallow is on top)
      drawResCurve(resDeepKey, trackConfig.res.color)   // Blue
      drawResCurve(resMedKey, trackConfig.res.medColor) // Red
      drawResCurve(resShalKey, trackConfig.res.shalColor) // Orange
      
      xOffset += trackConfig.res.width
    }
    
    // ========================================
    // DENSITY-NEUTRON TRACK
    // ========================================
    const densKey = mapping.DENS
    const neutKey = mapping.NEUT
    
    if ((densKey && curves[densKey]) || (neutKey && curves[neutKey])) {
      const dnGroup = svg.append('g')
        .attr('class', 'track track-dn')
      
      // Header
      dnGroup.append('rect')
        .attr('x', xOffset)
        .attr('y', 0)
        .attr('width', trackConfig.dn.width)
        .attr('height', headerHeight)
        .attr('fill', '#F0F0F0')
        .attr('stroke', '#808080')
      
      // Two sub-headers
      dnGroup.append('rect')
        .attr('x', xOffset + 2)
        .attr('y', 2)
        .attr('width', trackConfig.dn.width - 4)
        .attr('height', 16)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      dnGroup.append('text')
        .attr('x', xOffset + trackConfig.dn.width / 2)
        .attr('y', 13)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('font-weight', 'bold')
        .text('Bulk Density / Neutron Porosity')
      
      dnGroup.append('rect')
        .attr('x', xOffset + 2)
        .attr('y', 20)
        .attr('width', trackConfig.dn.width - 4)
        .attr('height', 16)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      dnGroup.append('text')
        .attr('x', xOffset + trackConfig.dn.width / 2)
        .attr('y', 31)
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .text(`${trackConfig.dn.densMin} ${trackConfig.dn.densUnit} ${trackConfig.dn.densMax}  |  ${trackConfig.dn.neutMax} ${trackConfig.dn.neutUnit} ${trackConfig.dn.neutMin}`)
      
      drawTrackBackground(dnGroup, xOffset, trackConfig.dn.width, headerHeight, plotHeight)
      
      // Density scale
      const densScale = d3.scaleLinear()
        .domain([trackConfig.dn.densMin, trackConfig.dn.densMax])
        .range([xOffset, xOffset + trackConfig.dn.width])
      
      // Neutron scale (reversed)
      const neutScale = d3.scaleLinear()
        .domain([trackConfig.dn.neutMax, trackConfig.dn.neutMin])
        .range([xOffset, xOffset + trackConfig.dn.width])
      
      // Density
      if (densKey && curves[densKey]) {
        const densData = depth.map((d, i) => ({
          depth: d,
          value: curves[densKey][i]
        }))
        
        const densLine = createLine(densScale)
        dnGroup.append('path')
          .datum(densData)
          .attr('d', densLine)
          .attr('fill', 'none')
          .attr('stroke', trackConfig.dn.densColor)
          .attr('stroke-width', 1.5)
      }
      
      // Neutron
      if (neutKey && curves[neutKey]) {
        const neutData = depth.map((d, i) => ({
          depth: d,
          value: curves[neutKey][i]
        }))
        
        const neutLine = createLine(neutScale)
        dnGroup.append('path')
          .datum(neutData)
          .attr('d', neutLine)
          .attr('fill', 'none')
          .attr('stroke', trackConfig.dn.neutColor)
          .attr('stroke-dasharray', '4,2')
          .attr('stroke-width', 1.5)
      }

      // Crossover Shading
      if (settings.showDnCrossover && densKey && neutKey && curves[densKey] && curves[neutKey]) {
        // Create combined data for area generation
        const areaData = depth.map((d, i) => ({
          depth: d,
          dens: curves[densKey][i],
          neut: curves[neutKey][i],
          y: yScale(d)
        })).filter(d => 
          d.dens !== null && !isNaN(d.dens) && 
          d.neut !== null && !isNaN(d.neut)
        )

        // Gas Effect (Crossover): Density (Red) is to the LEFT of Neutron (Blue)
        // Visually: Density < Neutron (technically High Porosity vs Low Density)
        // Wait, Density Scale: 1.95 -> 2.95 (Left to Right). Low Density = Left.
        // Neutron Scale: 0.45 -> -0.15 (Left to Right). High Porosity = Left.
        // Real Gas Crossover: Density reads LOW (Left), Neutron reads LOW (Right).
        // So Density X < Neutron X.
        
        // Define Gas Area (fill Yellow or Red)
        const gasArea = d3.area()
          .y(d => d.y)
          .x0(d => densScale(d.dens))
          .x1(d => neutScale(d.neut))
          .defined(d => densScale(d.dens) < neutScale(d.neut)) // Only where Density is Left of Neutron

        dnGroup.insert('path', ':first-child')
          .datum(areaData)
          .attr('d', gasArea)
          .attr('fill', '#FFFF00') // Yellow for Gas
          .attr('fill-opacity', 0.5)

        // Separation (Shale/Water): Density is to the RIGHT of Neutron
        const separationArea = d3.area()
          .y(d => d.y)
          .x0(d => densScale(d.dens))
          .x1(d => neutScale(d.neut))
          .defined(d => densScale(d.dens) > neutScale(d.neut))
        
        dnGroup.insert('path', ':first-child')
          .datum(areaData)
          .attr('d', separationArea)
          .attr('fill', '#808080') // Gray for Shale/Water separation
          .attr('fill-opacity', 0.3)
      }
      
      xOffset += trackConfig.dn.width
    }
    
    // ========================================
    // CADE TRACK REMOVED
    // ========================================
    
  }, [curveData, wellData, settings, dimensions, trackConfig])
  
  if (loading) {
    return (
      <div className="log-viewer loading">
        <div className="loading-spinner"></div>
        <span>Loading log data...</span>
      </div>
    )
  }
  
  if (!curveData) {
    return (
      <div className="log-viewer empty">
        <span>No data to display</span>
      </div>
    )
  }
  
  return (
    <div className="log-viewer" ref={containerRef}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ fontFamily: 'Segoe UI, Tahoma, sans-serif' }}
      />
    </div>
  )
}

export default LogViewer
