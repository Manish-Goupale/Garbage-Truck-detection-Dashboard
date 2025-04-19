<?php
// Enable error reporting
error_reporting(E_ALL);
ini_set('display_errors', 1);

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // For CORS

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "user_database";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die(json_encode(["error" => "Connection failed: " . $conn->connect_error]));
}

// Insert Truck Data from Dashboard
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['truck_number']) && isset($_POST['status'])) {
    $truck_number = $conn->real_escape_string($_POST['truck_number']);
    $status = $conn->real_escape_string($_POST['status']);

    $insert_sql = "INSERT INTO trucks (truck_number, status) VALUES ('$truck_number', '$status')";

    if ($conn->query($insert_sql) === TRUE) {
        echo json_encode(["message" => "Truck data inserted successfully!"]);
    } else {
        echo json_encode(["error" => "Error inserting data: " . $conn->error]);
    }
} else {
    echo json_encode(["error" => "Invalid request."]);
}

$conn->close();
?>
