<?php
// Enable error reporting
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Database connection details
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "user_database";

// Create connection
$conn = new mysqli($servername, $username, $password);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Create database if it doesn't exist
$sql = "CREATE DATABASE IF NOT EXISTS $dbname";
if ($conn->query($sql) !== TRUE) {
    die("Error creating database: " . $conn->error);
}

// Select the database
$conn->select_db($dbname);

// Create users table if it doesn't exist
$sql = "CREATE TABLE IF NOT EXISTS users (
    id INT(11) AUTO_INCREMENT PRIMARY KEY,
    plantName VARCHAR(255) NOT NULL,
    mineName VARCHAR(255) NOT NULL,
    truckCategory VARCHAR(50) NOT NULL,
    userID VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    mobile VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)";

if ($conn->query($sql) !== TRUE) {
    die("Error creating table: " . $conn->error);
}

// Registration logic
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['register'])) {
    $plantName = $conn->real_escape_string($_POST['plantName']);
    $mineName = $conn->real_escape_string($_POST['mineName']);
    $truckCategory = $conn->real_escape_string($_POST['truckCategory']);
    $userID = $conn->real_escape_string($_POST['userID']);
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT); // Encrypt password
    $email = $conn->real_escape_string($_POST['email']);
    $mobile = $conn->real_escape_string($_POST['mobile']);
    
    // Check if userID already exists
    $check_sql = "SELECT * FROM users WHERE userID = '$userID'";
    $result = $conn->query($check_sql);
    
    if ($result->num_rows > 0) {
        echo "User ID already exists. Please choose a different User ID.";
    } else {
        // Insert the user data into the database
        $sql = "INSERT INTO users (plantName, mineName, truckCategory, userID, password, email, mobile)
                VALUES ('$plantName', '$mineName', '$truckCategory', '$userID', '$password', '$email', '$mobile')";

        if ($conn->query($sql) === TRUE) {
            echo "Registration successful! You can now login using your User ID and password.";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
}

// Login logic
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['login'])) {
    $userID = $conn->real_escape_string($_POST['loginUserID']);
    $password = $_POST['loginPassword'];

    // Fetch user data from the database
    $sql = "SELECT * FROM users WHERE userID='$userID'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();

        // Verify password using password_verify() for hashed password
        if (password_verify($password, $row['password'])) {
            // Successful login
            echo "Login successful";
            // You could also start a session here and store user data
            // session_start();
            // $_SESSION['user_id'] = $row['id'];
            // $_SESSION['user_name'] = $row['userID'];
        } else {
            echo "Invalid password. Please try again.";
        }
    } else {
        echo "No user found with this User ID. Please register first.";
    }
}

// Close the connection
$conn->close();
?>