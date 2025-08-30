require("dotenv").config();

const express = require("express");
const { spawn } = require("child_process");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const bcrypt = require("bcrypt");
const session = require("express-session");
const mongoose = require("mongoose");
const app = express();
const port = 3000;

// Connect to MongoDB
mongoose
  .connect(
    process.env.MONGODB_URI ||
      "mongodb+srv://23f3002157:QWERTYUIOP_1234567890@inceptrixcluster.ejhly36.mongodb.net/?retryWrites=true&w=majority&appName=InceptrixCluster",
    {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    }
  )
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB connection error:", err));

// User Schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
});

const User = mongoose.model("User", userSchema);

const GEMINI_API_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const extension = path.extname(file.originalname);
    cb(null, file.fieldname + "-" + Date.now() + extension);
  },
});

const upload = multer({ storage: storage });

app.set("view engine", "ejs");

app.use(express.static("public"));
app.use("/eeg_data", express.static(path.join(__dirname, "EEG_Data_new")));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Session middleware
app.use(
  session({
    secret: process.env.SESSION_SECRET || "eeg_analysis_secret_key",
    resave: false,
    saveUninitialized: false,
    cookie: {
      secure: process.env.NODE_ENV === "production",
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
    },
  })
);

// Authentication middleware
const isAuthenticated = (req, res, next) => {
  if (req.session && req.session.userId) {
    return next();
  }
  res.redirect("/");
};

// Routes
app.get("/", (req, res) => {
  if (req.session.userId) {
    return res.redirect("/dashboard");
  }
  res.render("login");
});

app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user by email
    const user = await User.findOne({ email });

    if (!user) {
      return res
        .status(400)
        .json({ success: false, message: "Invalid email or password" });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res
        .status(400)
        .json({ success: false, message: "Invalid email or password" });
    }

    // Set session
    req.session.userId = user._id;
    req.session.userName = user.name;

    return res.json({ success: true, message: "Login successful" });
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({ success: false, message: "Server error" });
  }
});

app.post("/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res
        .status(400)
        .json({ success: false, message: "Email already in use" });
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create new user
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
    });

    await newUser.save();

    return res.json({ success: true, message: "User registered successfully" });
  } catch (error) {
    console.error("Signup error:", error);
    return res.status(500).json({ success: false, message: "Server error" });
  }
});

app.get("/logout", (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return console.error("Logout error:", err);
    }
    res.redirect("/");
  });
});

app.get("/dashboard", isAuthenticated, (req, res) => {
  res.render("dashboard", {
    userName: req.session.userName,
  });
});

app.get("/analyze", isAuthenticated, (req, res) => {
  res.render("analyze", {
    userName: req.session.userName,
  });
});

app.post("/analyze", isAuthenticated, (req, res) => {
  if (!req.body || !req.body.patientId) {
    return res.status(400).render("error", {
      error: "Missing required parameter",
      details: "Patient ID is required",
    });
  }

  const patientId = req.body.patientId;
  const duration = req.body.duration || 10;

  const sessionId = Date.now();
  const outputDir = path.join(__dirname, "EEG_Data_new");

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log(
    `Starting EEG analysis for patient ${patientId} for ${duration} seconds`
  );

  const pythonProcess = spawn("python", [
    path.join(__dirname, "python", "eeg_processor.py"),
    patientId,
    duration,
  ]);

  let pythonData = "";
  let pythonError = "";

  pythonProcess.stdout.on("data", (data) => {
    pythonData += data.toString();
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    pythonError += data.toString();
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);

    if (code !== 0) {
      return res.render("error", {
        error: "Python script execution failed",
        details: pythonError,
      });
    }

    const files = fs.readdirSync(outputDir);
    const imageFiles = files.filter(
      (file) => file.includes(`eeg_plot_${patientId}`) && file.endsWith(".png")
    );

    imageFiles.sort((a, b) => {
      return (
        fs.statSync(path.join(outputDir, b)).mtime.getTime() -
        fs.statSync(path.join(outputDir, a)).mtime.getTime()
      );
    });

    const edfFiles = files.filter(
      (file) => file.includes(`eeg_${patientId}`) && file.endsWith(".edf")
    );

    edfFiles.sort((a, b) => {
      return (
        fs.statSync(path.join(outputDir, b)).mtime.getTime() -
        fs.statSync(path.join(outputDir, a)).mtime.getTime()
      );
    });

    const reportFiles = files.filter(
      (file) =>
        file.includes(`eeg_analysis_${patientId}`) && file.endsWith(".txt")
    );

    reportFiles.sort((a, b) => {
      return (
        fs.statSync(path.join(outputDir, b)).mtime.getTime() -
        fs.statSync(path.join(outputDir, a)).mtime.getTime()
      );
    });

    let reportContent = "";
    if (reportFiles.length > 0) {
      reportContent = fs.readFileSync(
        path.join(outputDir, reportFiles[0]),
        "utf8"
      );
    }

    res.render("results", {
      patientId: patientId,
      duration: duration,
      imageFile: imageFiles.length > 0 ? imageFiles[0] : null,
      edfFile: edfFiles.length > 0 ? edfFiles[0] : null,
      reportContent: reportContent,
      pythonOutput: pythonData,
      userName: req.session.userName,
    });
  });
});

app.get("/predict", isAuthenticated, (req, res) => {
  res.render("predict", {
    userName: req.session.userName,
  });
});

app.post("/predict", isAuthenticated, upload.single("edfFile"), (req, res) => {
  if (!req.file) {
    return res.render("error", {
      error: "No file uploaded",
      details: "Please upload an EDF file",
      userName: req.session.userName,
    });
  }

  const edfFilePath = req.file.path;
  const outputDir = path.join(__dirname, "EEG_Data_new");

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log(`Starting seizure prediction for file: ${req.file.originalname}`);

  const pythonProcess = spawn("python", [
    path.join(__dirname, "python", "predict_seizure.py"),
    edfFilePath,
    outputDir,
  ]);

  let pythonData = "";
  let pythonError = "";

  pythonProcess.stdout.on("data", (data) => {
    pythonData += data.toString();
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    pythonError += data.toString();
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);

    if (code !== 0) {
      return res.render("error", {
        error: "Python script execution failed",
        details: pythonError,
        userName: req.session.userName,
      });
    }

    let prediction = "Unknown";
    let confidence = "0%";
    let visualizationFile = "";

    try {
      const outputLines = pythonData.split("\n");
      for (const line of outputLines) {
        if (line.includes("Result:")) {
          prediction = line.split("Result:")[1].trim();
        } else if (line.includes("Confidence:")) {
          confidence = line.split("Confidence:")[1].trim();
        } else if (line.includes("Visualization saved to:")) {
          const fullPath = line.split("Visualization saved to:")[1].trim();
          visualizationFile = path.basename(fullPath);
        }
      }
    } catch (err) {
      console.error("Error parsing Python output:", err);
    }

    res.render("prediction_results", {
      fileName: req.file.originalname,
      prediction: prediction,
      confidence: confidence,
      visualizationFile: visualizationFile,
      pythonOutput: pythonData,
      userName: req.session.userName,
    });
  });
});

app.post("/chat", async (req, res) => {
  const { userInput, pageType, eegData } = req.body;
  let prompt = "";

  if (pageType === "home") {
    prompt = "You are an EEG specialist. Answer general EEG-related queries.";
  } else if (pageType === "seizure") {
    prompt =
      "You are a Seizure Specialist. Provide expert medical guidance on seizures.";
  } else if (pageType === "recording") {
    prompt = `You are an EEG data analyst. Analyze the given EEG data and provide meaningful insights. Here is the EEG data: ${eegData}`;
  }

  try {
    console.log(`Sending request to Gemini API: ${GEMINI_API_URL}`);

    const requestData = {
      contents: [
        {
          role: "user",
          parts: [{ text: `${prompt}\nUser: ${userInput}` }],
        },
      ],
    };

    console.log("Request payload:", JSON.stringify(requestData));

    const response = await axios.post(
      `${GEMINI_API_URL}?key=${GEMINI_API_KEY}`,
      requestData
    );

    console.log("Response received:", response.status);

    if (
      response.data &&
      response.data.candidates &&
      response.data.candidates[0] &&
      response.data.candidates[0].content &&
      response.data.candidates[0].content.parts &&
      response.data.candidates[0].content.parts[0]
    ) {
      res.json({ reply: response.data.candidates[0].content.parts[0].text });
    } else {
      console.error("Unexpected response structure:", response.data);
      res
        .status(500)
        .json({ error: "Invalid response structure from Gemini AI" });
    }
  } catch (error) {
    console.error("Error with Gemini API:", error.message);

    if (error.response) {
      console.error("Response data:", error.response.data);
      console.error("Response status:", error.response.status);
    }

    res.status(500).json({ error: "Failed to get response from Gemini AI" });
  }
});

// Handle 404
app.use((req, res) => {
  res.status(404).render("error", {
    error: "Page Not Found",
    details: "The page you are looking for does not exist.",
    userName: req.session.userId ? req.session.userName : null,
  });
});

// Start the server
app.listen(port, () => {
  console.log(`EEG Analysis app listening at http://localhost:${port}`);
});
