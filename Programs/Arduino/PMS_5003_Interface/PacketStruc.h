#define PACKET_SIZE  32 

//--- PACKET STRUCTURE ---//
// Constant values
#define Start_Char1 0
#define Start_Char2 1
#define FrameLen_High 2
#define FrameLen_Low 3
// PM (10->1.0) concentration unit μ g/m3（CF=1，standard particle)
#define PM10std_High 4
#define PM10std_Low 5
#define PM25std_High 6
#define PM25std_Low 7
#define PM100std_High 8
#define PM100std_Low 9
// PM (10->1.0) concentration unit μ g/m3（under atmospheric environment）
#define PM10atm_High 10
#define PM10atm_Low 11
#define PM25atm_High 12
#define PM25atm_Low 13
#define PM100atm_High 14
#define PM100atm_Low 15
// Number of particles with diameter beyond (03->0.3) um
#define Count03_High 16
#define Count03_Low 17
#define Count05_High 18
#define Count05_Low 19
#define Count10_High 20
#define Count10_Low 21
#define Count25_High 22
#define Count25_Low 23
#define Count50_High 24
#define Count50_Low 25
#define Count100_High 26
#define Count100_Low 27
// Reserved
#define Res_High 28
#define Res_Low 29
// Checksum
#define Checksum_High 30
#define Checksum_Low 31