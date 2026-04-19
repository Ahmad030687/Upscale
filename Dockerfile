# 1. Base image (Python 3.10 slim version for better performance)
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. System dependencies install karein (OpenCV aur Models ke liye zaroori hain)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Pip, Setuptools aur Wheel ko upgrade karein
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 5. Core AI Libraries pehle install karein (Taki basicsr crash na kare)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. BasicSR Patch Installation (Yehi main fix hai)
# Hum basicsr ko bina dependencies ke install kar rahe hain taki setup.py error na de
RUN pip install --no-cache-dir basicsr==1.4.2 --no-deps

# 7. Baqi sari requirements install karein
# Note: requirements.txt mein basicsr nahi likhna ab
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8. Project files copy karein
COPY . .

# 9. API Port expose karein
EXPOSE 7860

# 10. Run the API
CMD ["python", "app.py"]
