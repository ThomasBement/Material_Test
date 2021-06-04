#include <SoftwareSerial.h>
#include "PacketStruc.h"
SoftwareSerial pmsSerial(2, 3);

// Macros
#define MAKE_INT(hi,low) (((unsigned char)(hi)<<8)+(unsigned char)(low))
#define DELAY_PERIOD 100

uint8_t  buffer[64];
uint16_t calculated_checksum;
uint16_t pkt_checksum;
uint16_t priv_pkt_checksum;
uint16_t  pass_count;
volatile double CalibrationValue;
unsigned long time;

void setup() {
  // our debugging output
  Serial.begin(9600);
 
  // sensor baud rate is 9600
  pmsSerial.begin(9600);

  // write unique information to check buffer
  buffer[32] = 0xAA;
}

void loop() {

  if (buffer[32] != (uint8_t)0xAA){
    Serial.println("BUFFER OVERWRITTEN");
    Serial.print("Expected 0xAA, but got 0x");Serial.println(buffer[32], HEX);
    while(1);
  }
  
  // Check data packet function
  if (!readPMSdata) return;

  // Read data packet and output serial data condition
  else{
    // Compute time in seconds from power on
    time = millis();
    // Read pms data
    pmsSerial.readBytes(buffer, PACKET_SIZE);

    // Serial outputs

    // Time
    Serial.print(time);Serial.print(" ");
    // Particles under 0.3 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count03_High],buffer[Count03_Low]));Serial.print(" ");
    // Particles under 0.5 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count05_High],buffer[Count05_Low]));Serial.print(" ");
    // Particles under 1.0 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count10_High],buffer[Count10_Low]));Serial.print(" ");
    // Particles under 2.5 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count25_High],buffer[Count25_Low]));Serial.print(" ");
    // Particles under 5.0 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count50_High],buffer[Count50_Low]));Serial.print(" ");
    // Particles under 10 um
    Serial.print((uint16_t)MAKE_INT(buffer[Count100_High],buffer[Count100_Low]));Serial.print(" ");
    // PM VALUES
    // PM 1.0 Standard Particle
    Serial.print((uint16_t)MAKE_INT(buffer[PM10std_High],buffer[PM10std_Low]));Serial.print(" ");
    // PM 2.5 Standard Particle
    Serial.print((uint16_t)MAKE_INT(buffer[PM25std_High],buffer[PM25std_Low]));Serial.print(" ");
    // PM 10.0 Standard Particle
    Serial.print((uint16_t)MAKE_INT(buffer[PM100std_High],buffer[PM100std_Low]));Serial.print(" ");
    // PM 1.0 Atmospheric Environment
    Serial.print((uint16_t)MAKE_INT(buffer[PM10atm_High],buffer[PM10atm_Low]));Serial.print(" ");
    // PM 1.0 Atmospheric Environment
    Serial.print((uint16_t)MAKE_INT(buffer[PM25atm_High],buffer[PM25atm_Low]));Serial.print(" ");
    // PM 1.0 Atmospheric Environment
    Serial.print((uint16_t)MAKE_INT(buffer[PM100atm_High],buffer[PM100atm_Low]));Serial.println();
    pass_count +=1;
  }
}

boolean readPMSdata(void) {
  if (!pmsSerial.available()) return false;
  
  // Read untill 0x42 start byte
  if ((uint8_t)pmsSerial.peek() != 0x42) {
    pmsSerial.read();
    return false;
  }
 
  // Now read all 32 bytes
  if (pmsSerial.available() < PACKET_SIZE) {
    return false;
  }

  pmsSerial.readBytes(buffer, PACKET_SIZE);
  
  // Calculate check sum excluding checksum from buffer
  calculated_checksum = 0;
  for (uint8_t i=0; i<PACKET_SIZE-3; i++) {
    calculated_checksum += (unsigned char)buffer[i];
  }
  
  // Find buffer checksum (packet size is 32)
  pkt_checksum = (uint16_t)MAKE_INT(buffer[PACKET_SIZE-2],buffer[PACKET_SIZE-1]);

  if ((buffer[3]!=0x1C)||(buffer[2]!=0x0)){
    //Serial.println("BUFFER HAS UNEXPECTED LENGTH");
    return false;
  }
  
  if (pkt_checksum!=calculated_checksum){
    for (uint8_t j=0; j<PACKET_SIZE; j++) {
      Serial.print("0x"); Serial.print((unsigned char)buffer[j], HEX); Serial.print(", ");
    }
    Serial.println();
    Serial.println("CHECK SUM ERROR:");
    Serial.print("calculated_checksum:");Serial.println((uint16_t)calculated_checksum, HEX);
    Serial.print("pkt_checksum:");Serial.println((uint16_t)pkt_checksum, HEX);
    return false;
  }

  // Check if current checksum is the same as the last one if so
  if (pkt_checksum==priv_pkt_checksum) return false; 
  
  else {
    // Update privious check sum after output
    priv_pkt_checksum = pkt_checksum;
    return true;
  }
}
