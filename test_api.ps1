$headers = @{ "Content-Type" = "application/json" }
$body = @{
    phone_number = "+917418933402"
    latitude = 12.9716
    longitude = 77.5946
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/register_phone" -Method POST -Headers $headers -Body $body
