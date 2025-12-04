document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
        });

// Global fallback for API_URL to avoid ReferenceError in any cached handlers
if (typeof window !== 'undefined') {
    window.API_URL = window.API_URL || 'http://localhost:8000/deblur';
    try {
        if (typeof API_URL === 'undefined') {
            var API_URL = window.API_URL;
        }
    } catch (e) {
        // no-op
    }
}
    }

// Extreme pipeline: directional iterative sharpening approximating motion deblur
async function processExtremeSharpen(imgEl, { amount = 2.5, radius = 3.2, lce = true, length = 18, angle = 0, iters = 12 } = {}) {
    // Downscale for speed if needed
    if (!imgEl.complete) {
        await new Promise((res) => imgEl.addEventListener('load', res, { once: true }));
    }
    const srcW = imgEl.naturalWidth || imgEl.width;
    const srcH = imgEl.naturalHeight || imgEl.height;
    const maxSide = 1024;
    const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w; canvas.height = h;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(imgEl, 0, 0, w, h);

    // Iterative directional enhancement
    for (let i = 0; i < iters; i++) {
        // Base unsharp
        await applyUnsharpInPlace(canvas, { radius: Math.max(0.8, radius * 0.7), amount: amount * 0.6, threshold: 0 });

        // Directional sharpen: rotate, unsharp, rotate back
        const dirAmount = amount * 0.7;
        const dirRadius = Math.max(0.6, Math.min(6, length / 6));
        await applyDirectionalSharpen(canvas, angle, dirRadius, dirAmount);

        // Mild dehaze-like local contrast
        if (lce && (i % 3 === 0)) {
            await applyUnsharpInPlace(canvas, { radius: 10, amount: 0.28, threshold: 0 });
        }
    }

    // Fine detail pass at the end
    await applyUnsharpInPlace(canvas, { radius: 0.9, amount: Math.min(1.2, amount * 0.5), threshold: 0 });

    return canvas.toDataURL('image/png');
}

async function applyDirectionalSharpen(canvas, angleDeg, radius, amount) {
    const w = canvas.width, h = canvas.height;
    const temp = document.createElement('canvas');
    temp.width = w; temp.height = h;
    const tctx = temp.getContext('2d');
    tctx.save();
    tctx.translate(w/2, h/2);
    tctx.rotate((angleDeg * Math.PI)/180);
    tctx.translate(-w/2, -h/2);
    tctx.drawImage(canvas, 0, 0);
    tctx.restore();

    await applyUnsharpInPlace(temp, { radius: radius, amount: amount, threshold: 0 });

    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.translate(w/2, h/2);
    ctx.rotate(-(angleDeg * Math.PI)/180);
    ctx.translate(-w/2, -h/2);
    ctx.drawImage(temp, 0, 0);
    ctx.restore();
}

// Stronger processing pipeline for visible difference
async function processStrongSharpen(imgEl, { amount = 2.0, radius = 2.8, lce = true } = {}) {
    // Draw original to canvas
    if (!imgEl.complete) {
        await new Promise((res) => imgEl.addEventListener('load', res, { once: true }));
    }
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w; canvas.height = h;
    ctx.drawImage(imgEl, 0, 0, w, h);

    // Pass 1: strong edge sharpening
    await applyUnsharpInPlace(canvas, { radius: Math.max(0.5, radius), amount: 1.5 * amount, threshold: 0 });
    // Pass 2: fine detail
    await applyUnsharpInPlace(canvas, { radius: Math.max(0.5, radius * 0.5), amount: 0.6 * amount, threshold: 0 });
    // Pass 3: local contrast with large radius and mild amount
    if (lce) {
        await applyUnsharpInPlace(canvas, { radius: 12.0, amount: 0.35, threshold: 0 });
    }

    return canvas.toDataURL('image/png');
}

async function applyUnsharpInPlace(canvas, opts) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const src = ctx.getImageData(0, 0, w, h);
    const blurred = gaussianBlurSeparable(src, w, h, opts.radius);
    const s = src.data;
    const b = blurred.data;
    const out = ctx.createImageData(w, h);
    const d = out.data;
    const amount = opts.amount ?? 1.5;
    const thr = opts.threshold ?? 0;

    for (let i = 0; i < s.length; i += 4) {
        const dr = s[i]   - b[i];
        const dg = s[i+1] - b[i+1];
        const db = s[i+2] - b[i+2];
        const ar = Math.abs(dr) > thr ? dr : 0;
        const ag = Math.abs(dg) > thr ? dg : 0;
        const ab = Math.abs(db) > thr ? db : 0;
        d[i]   = clamp255(s[i]   + amount * ar);
        d[i+1] = clamp255(s[i+1] + amount * ag);
        d[i+2] = clamp255(s[i+2] + amount * ab);
        d[i+3] = 255;
    }
    ctx.putImageData(out, 0, 0);
}

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                // Close mobile menu if open
                if (!mobileMenu.classList.contains('hidden')) {
                    mobileMenu.classList.add('hidden');
                }
                
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust for fixed header
                    behavior: 'smooth'
                });
            }
        });
    });

    // Image upload and preview functionality
    const uploadContainer = document.getElementById('upload-container');
    const imageUpload = document.getElementById('image-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const previewImage = document.getElementById('preview-image');
    const resultImage = document.getElementById('result-image');
    const placeholder = document.getElementById('placeholder');
    const beforeLabel = document.getElementById('before-label');
    const afterLabel = document.getElementById('after-label');
    const processBtn = document.getElementById('process-btn');
    const downloadBtn = document.getElementById('download-btn');
    const loading = document.getElementById('loading');
    const procControls = document.getElementById('proc-controls');
    const amountRange = document.getElementById('amount-range');
    const radiusRange = document.getElementById('radius-range');
    const lceCheck = document.getElementById('lce-check');
    const modeSelect = document.getElementById('mode-select');
    const deconvControls = document.getElementById('deconv-controls');
    const psfLen = document.getElementById('psf-length');
    const psfAngle = document.getElementById('psf-angle');
    const rlIters = document.getElementById('rl-iters');
    let processedDataUrl = '';
    window.API_URL = 'http://localhost:8000/deblur';

    // Wire the explicit upload button
    if (uploadBtn) {
        uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageUpload.click();
        });
    }

    // Handle click on upload container
    uploadContainer.addEventListener('click', () => {
        imageUpload.click();
    });

    // Handle file selection
    imageUpload.addEventListener('change', handleFileSelect);

    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadContainer.classList.add('border-blue-500', 'bg-blue-50');
    }

    function unhighlight() {
        uploadContainer.classList.remove('border-blue-500', 'bg-blue-50');
    }

    // Handle dropped files
    uploadContainer.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                alert('Please select an image file (JPG, PNG)');
                return;
            }
            
            // Check file size (5MB max)
            if (file.size > 5 * 1024 * 1024) {
                alert('File size should be less than 5MB');
                return;
            }
            
            // Display preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.classList.remove('hidden');
                placeholder.classList.add('hidden');
                processBtn.classList.remove('hidden');
                processBtn.disabled = false;
                // Show labels and controls
                if (beforeLabel) beforeLabel.classList.remove('hidden');
                if (afterLabel) afterLabel.classList.add('hidden');
                if (procControls) procControls.classList.remove('hidden');
                // Reset any previous result
                processedDataUrl = '';
                resultImage.classList.add('hidden');
                resultImage.src = '';
                
                // Reset download button if it was shown before
                downloadBtn.classList.add('hidden');
                downloadBtn.removeAttribute('href');
                downloadBtn.removeAttribute('download');
            };
            reader.readAsDataURL(file);
        }
    }

    // Process button click handler
    processBtn.addEventListener('click', processImage);

    async function processImage() {
        if (!previewImage.src) return;
        
        // Show loading state
        loading.classList.remove('hidden');
        processBtn.disabled = true;
        // Add processing visual effect to the preview image
        previewImage.classList.add('pre-unblur');
        
        try {
            // Simulate processing time
            await new Promise(resolve => setTimeout(resolve, 600));
            // Choose processing mode
            const amountVal = amountRange ? parseFloat(amountRange.value) : 2.0;
            const radiusVal = radiusRange ? parseFloat(radiusRange.value) : 2.8;
            const lceVal = lceCheck ? !!lceCheck.checked : true;
            const mode = modeSelect ? modeSelect.value : 'standard';
            if (mode === 'extreme') {
                const lenVal = psfLen ? parseInt(psfLen.value, 10) : 18;
                const angVal = psfAngle ? parseFloat(psfAngle.value) : 0;
                const itVal = rlIters ? parseInt(rlIters.value, 10) : 12;
                processedDataUrl = await processExtremeSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal, length: lenVal, angle: angVal, iters: itVal });
            } else if (mode === 'auto') {
                const est = await estimateMotionParams(previewImage);
                if (psfLen) psfLen.value = est.length.toFixed(0);
                if (psfAngle) psfAngle.value = est.angle.toFixed(0);
                if (rlIters) rlIters.value = est.iters.toFixed(0);
                // Show deconv controls in auto as well
                if (deconvControls) deconvControls.style.display = '';
                processedDataUrl = await processExtremeSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal, length: est.length, angle: est.angle, iters: est.iters });
            } else if (mode === 'model') {
                const file = imageUpload.files && imageUpload.files[0]
                    ? imageUpload.files[0]
                    : await dataUrlToFile(previewImage.src, 'upload.png');
                const apiResult = await deblurViaApi(file);
                processedDataUrl = `data:image/png;base64,${apiResult.image}`;
            } else {
                processedDataUrl = await processStrongSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal });
            }
            showSuccess();
        } catch (error) {
            console.error('Error processing image:', error);
            alert('An error occurred while processing the image. Please try again.');
            loading.classList.add('hidden');
            processBtn.disabled = false;
            previewImage.classList.remove('pre-unblur');
        }
    }
    
    function showSuccess() {
        loading.classList.add('hidden');
        previewImage.classList.remove('pre-unblur');
        
        // Show processed image inline with unblur animation
        if (processedDataUrl) {
            resultImage.src = processedDataUrl;
            resultImage.classList.remove('hidden');
            // Distinguish visually
            previewImage.classList.add('ring-2','ring-gray-300');
            resultImage.classList.add('ring-2','ring-green-400');
            // Subtle global boost for visibility
            resultImage.style.filter = 'contrast(1.08) saturate(1.06)';
            // Restart animation by toggling class
            resultImage.classList.remove('unblur-animate');
            // Force reflow
            void resultImage.offsetWidth;
            resultImage.classList.add('unblur-animate');
            if (afterLabel) afterLabel.classList.remove('hidden');
        }

        // Prepare download of the processed image
        downloadBtn.href = processedDataUrl || previewImage.src;
        downloadBtn.download = 'deblurred_' + (imageUpload.files[0]?.name || 'image.png');
        downloadBtn.classList.remove('hidden');
        
        // Show success message
        const successMessage = document.createElement('div');
        successMessage.className = 'mt-4 p-3 bg-green-100 text-green-700 rounded-md text-sm';
        successMessage.textContent = 'Image processed successfully! View result above or click Download to save it.';
        downloadBtn.parentNode.insertBefore(successMessage, downloadBtn.nextSibling);
        
        // Remove the success message after 5 seconds
        setTimeout(() => {
            successMessage.remove();
        }, 5000);
    }

    // Animate elements when they come into view
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('animate-fade-in');
            }
        });
    };
    
    // Initial check for elements in viewport
    animateOnScroll();
    
    // Check for elements as user scrolls
    window.addEventListener('scroll', animateOnScroll);

    // Add animation class to sections
    document.querySelectorAll('section').forEach((section, index) => {
        // Add a small delay to each section for staggered animation
        section.style.animationDelay = `${index * 0.1}s`;
        section.classList.add('animate-on-scroll');
    });

// Add active class to current section in navigation
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('nav a');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (pageYOffset >= (sectionTop - sectionHeight / 3)) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('text-blue-300');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('text-blue-300');
        }
    });
});

// Unsharp Mask: sharpen using (original + amount * (original - gaussianBlur(original)))
async function unsharpMask(imgEl, { radius = 2.0, amount = 1.5, threshold = 0 } = {}) {
    // Ensure the image is loaded with natural dimensions
    if (!imgEl.complete) {
        await new Promise((res) => imgEl.addEventListener('load', res, { once: true }));
    }
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(imgEl, 0, 0, w, h);

    try {
        const srcData = ctx.getImageData(0, 0, w, h);
        const blurred = gaussianBlurSeparable(srcData, w, h, radius);
        const s = srcData.data;
        const b = blurred.data;
        const out = ctx.createImageData(w, h);
        const d = out.data;

        const thr = threshold;

        for (let i = 0; i < s.length; i += 4) {
            const dr = s[i]   - b[i];
            const dg = s[i+1] - b[i+1];
            const db = s[i+2] - b[i+2];

            // Threshold to avoid amplifying noise
            const ar = Math.abs(dr) > thr ? dr : 0;
            const ag = Math.abs(dg) > thr ? dg : 0;
            const ab = Math.abs(db) > thr ? db : 0;

            d[i]   = clamp255(s[i]   + amount * ar);
            d[i+1] = clamp255(s[i+1] + amount * ag);
            d[i+2] = clamp255(s[i+2] + amount * ab);
            d[i+3] = 255;
        }

        ctx.putImageData(out, 0, 0);
        return canvas.toDataURL('image/png');
    } catch (e) {
        // Fallback: if ImageData blocked for any reason, return original
        return canvas.toDataURL('image/png');
    }
}

function clamp255(v) {
    return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function gaussianKernel1D(sigma) {
    const radius = Math.max(1, Math.round(sigma * 3));
    const size = radius * 2 + 1;
    const kernel = new Float32Array(size);
    const sigma2 = sigma * sigma;
    let sum = 0;
    for (let i = -radius, j = 0; i <= radius; i++, j++) {
        const v = Math.exp(-(i*i) / (2 * sigma2));
        kernel[j] = v;
        sum += v;
    }
    // normalize
    for (let i = 0; i < size; i++) kernel[i] /= sum;
    return { kernel, radius };
}

function gaussianBlurSeparable(srcImageData, w, h, sigma) {
    const { kernel, radius } = gaussianKernel1D(sigma);
    const src = srcImageData.data;
    const tmp = new Uint8ClampedArray(src.length);
    const out = new Uint8ClampedArray(src.length);

    // Horizontal pass
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let r = 0, g = 0, b = 0, a = 0;
            for (let k = -radius; k <= radius; k++) {
                const xx = Math.min(w - 1, Math.max(0, x + k));
                const idx = (y * w + xx) * 4;
                const kval = kernel[k + radius];
                r += src[idx] * kval;
                g += src[idx + 1] * kval;
                b += src[idx + 2] * kval;
                a += src[idx + 3] * kval;
            }
            const di = (y * w + x) * 4;
            tmp[di] = r; tmp[di+1] = g; tmp[di+2] = b; tmp[di+3] = a;
        }
    }

    // Vertical pass
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
            let r = 0, g = 0, b = 0, a = 0;
            for (let k = -radius; k <= radius; k++) {
                const yy = Math.min(h - 1, Math.max(0, y + k));
                const idx = (yy * w + x) * 4;
                const kval = kernel[k + radius];
                r += tmp[idx] * kval;
                g += tmp[idx + 1] * kval;
                b += tmp[idx + 2] * kval;
                a += tmp[idx + 3] * kval;
            }
            const di = (y * w + x) * 4;
            out[di] = r; out[di+1] = g; out[di+2] = b; out[di+3] = a;
        }
    }

    const blurred = new ImageData(out, w, h);
    return blurred;
}

// Live reprocess when controls change (after first processing)
function hasProcessed() {
    return !!(resultImage && resultImage.src && !resultImage.classList.contains('hidden'));
}

async function reprocessWithCurrentSettings() {
    if (!previewImage.src) return;
    loading.classList.remove('hidden');
    try {
        const amountVal = amountRange ? parseFloat(amountRange.value) : 2.0;
        const radiusVal = radiusRange ? parseFloat(radiusRange.value) : 2.8;
        const lceVal = lceCheck ? !!lceCheck.checked : true;
        const mode = modeSelect ? modeSelect.value : 'standard';
        if (mode === 'extreme') {
            const lenVal = psfLen ? parseInt(psfLen.value, 10) : 18;
            const angVal = psfAngle ? parseFloat(psfAngle.value) : 0;
            const itVal = rlIters ? parseInt(rlIters.value, 10) : 12;
            processedDataUrl = await processExtremeSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal, length: lenVal, angle: angVal, iters: itVal });
        } else if (mode === 'auto') {
            const est = await estimateMotionParams(previewImage);
            if (psfLen) psfLen.value = est.length.toFixed(0);
            if (psfAngle) psfAngle.value = est.angle.toFixed(0);
            if (rlIters) rlIters.value = est.iters.toFixed(0);
            processedDataUrl = await processExtremeSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal, length: est.length, angle: est.angle, iters: est.iters });
        } else if (mode === 'model') {
            const file = imageUpload.files && imageUpload.files[0]
                ? imageUpload.files[0]
                : await dataUrlToFile(previewImage.src, 'upload.png');
            const apiResult = await deblurViaApi(file);
            processedDataUrl = `data:image/png;base64,${apiResult.image}`;
        } else {
            processedDataUrl = await processStrongSharpen(previewImage, { amount: amountVal, radius: radiusVal, lce: lceVal });
        }
        resultImage.src = processedDataUrl;
        downloadBtn.href = processedDataUrl;
    } finally {
        loading.classList.add('hidden');
    }
}

async function deblurViaApi(file) {
    const form = new FormData();
    form.append('file', file);
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(), 60000);
    try {
        const apiUrl = window.API_URL || 'http://localhost:8000/deblur';
        const resp = await fetch(apiUrl, { method: 'POST', body: form, signal: controller.signal });
        if (!resp.ok) {
            const t = await resp.text();
            throw new Error(`API error ${resp.status}: ${t}`);
        }
        const data = await resp.json();
        return data;
    } finally {
        clearTimeout(to);
    }
}

async function dataUrlToFile(dataUrl, filename) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || 'image/png' });
}

// Auto estimation: estimate motion angle and length from Sobel gradients and Laplacian stats
async function estimateMotionParams(imgEl) {
    if (!imgEl.complete) {
        await new Promise((res) => imgEl.addEventListener('load', res, { once: true }));
    }
    const maxSide = 640;
    const iw = imgEl.naturalWidth || imgEl.width;
    const ih = imgEl.naturalHeight || imgEl.height;
    const scale = Math.min(1, maxSide / Math.max(iw, ih));
    const w = Math.max(64, Math.round(iw * scale));
    const h = Math.max(64, Math.round(ih * scale));
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);

    // Grayscale
    const g = new Float32Array(w * h);
    const d = img.data;
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
        g[j] = 0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2];
    }

    // Sobel
    const gx = new Float32Array(w * h);
    const gy = new Float32Array(w * h);
    const sobelX = [-1,0,1,-2,0,2,-1,0,1];
    const sobelY = [-1,-2,-1,0,0,0,1,2,1];
    const k = 1; // radius
    for (let y = 1; y < h-1; y++) {
        for (let x = 1; x < w-1; x++) {
            let sx = 0, sy = 0;
            let idx = 0;
            for (let ky=-k; ky<=k; ky++) {
                for (let kx=-k; kx<=k; kx++) {
                    const px = x + kx, py = y + ky;
                    const val = g[py*w + px];
                    sx += val * sobelX[idx];
                    sy += val * sobelY[idx];
                    idx++;
                }
            }
            gx[y*w + x] = sx; gy[y*w + x] = sy;
        }
    }

    // Orientation histogram weighted by magnitude
    const bins = 180; // -90..+90
    const hist = new Float32Array(bins);
    const mags = new Float32Array(w*h);
    for (let i = 0; i < mags.length; i++) {
        const m = Math.hypot(gx[i], gy[i]);
        mags[i] = m;
        let ang = Math.atan2(gy[i], gx[i]) * 180 / Math.PI; // -180..180
        ang = ((ang % 180) + 180) % 180; // 0..180
        if (ang > 90) ang = 180 - ang;   // 0..90
        const bin = Math.min(bins-1, Math.max(0, Math.round((ang/90) * (bins-1))));
        hist[bin] += m;
    }
    // Dominant angle around 0 in [-45,45]
    let maxV = -1, maxI = 0;
    for (let i = 0; i < bins; i++) { if (hist[i] > maxV) { maxV = hist[i]; maxI = i; } }
    const angle = (maxI / (bins-1)) * 90 - 45;

    // Edge strength (top 5% magnitudes)
    const mCopy = Array.from(mags);
    mCopy.sort((a,b)=>a-b);
    const start = Math.floor(mCopy.length * 0.95);
    let sumTop = 0; let cnt = 0;
    for (let i = start; i < mCopy.length; i++) { sumTop += mCopy[i]; cnt++; }
    const topMean = cnt ? (sumTop / cnt) : 0;
    const norm = topMean / 255;
    // Map to length: lower norm -> longer blur
    let length = 6 + (1 - Math.min(1, Math.max(0, norm))) * 22; // 6..28
    // Iterations based on blur severity
    const iters = Math.round(10 + (1 - Math.min(1, Math.max(0, norm))) * 8); // 10..18

    return { angle: Math.max(-45, Math.min(45, angle)), length: Math.max(6, Math.min(28, length)), iters };
}

// Toggle deconvolution controls visibility by mode
if (modeSelect && deconvControls) {
    const updateDeconvVis = () => {
        deconvControls.style.display = (modeSelect.value === 'extreme' || modeSelect.value === 'auto') ? '' : 'none';
    };
    modeSelect.addEventListener('change', () => { updateDeconvVis(); if (hasProcessed()) reprocessWithCurrentSettings(); });
    updateDeconvVis();
}

;[amountRange, radiusRange, lceCheck, modeSelect, psfLen, psfAngle, rlIters].forEach(ctrl => {
    if (ctrl) {
        ctrl.addEventListener('input', () => {
            if (hasProcessed()) {
                reprocessWithCurrentSettings();
            }
        });
    }
});

});
