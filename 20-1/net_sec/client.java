import java.math.BigInteger;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.security.SecureRandom;

import javax.crypto.*;
import javax.crypto.spec.*;

import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Base64.Decoder;
import java.util.Base64.Encoder;

public class Mainclient {	
    AES aesOBJ;
    RSA rsaOBJ;

	public static void main(String[] args) throws Exception {		
		try {			
			Socket sock = new Socket("192.168.0.2", 9506);
            aesOBJ = new AES();
            rsaOBJ = new RSA();
            			
			BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(System.in));
			String receivestr;
			
			Sendthread sen_thread = new Sendthread();
			sen_thread.setSocket(sock);
			sen_thread.start();
            
            // receive_thread
			synchronized(sen_thread) {
				receivestr = tmpbuf.readLine();
                System.out.printf("Received Public Key : " + receivestr);
                //rsaOBJ.publickey = Base64.getDecoder().decode(receivestr);
                rsaOBJ.publickey = receivestr.getBytes();
				aesOBJ.createSecretkey();
				sen_thread.notify(); ////////////3
			}
			
			try {				
				while(true) {
                    receivestr = tmpbuf.readLine();

                    if(receivestr != null){
                        System.out.println("Received : " + aesOBJ.Decrpyt(receivestr));
                        System.out.println("Encrypted Message : " + receivestr);
                    }

                    if(receivestr.substring(1,4).equals("exit")) {
						break;
					}
				}
                
                tmpbuf.close();
                sock.close();
			} catch(IOException e) {
				e.printStackTrace();
            }
            
			System.out.println("Connection closed");
			sock.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
    }
    
    class Sendthread extends Thread{
        private Socket sock;
        
        @Override
        public void run() {
            super.run();
            try {
                BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(System.in));
                PrintWriter sendwriter = new PrintWriter(sock.getOutputStream());
                String sendstr, timestamp;
                
                synchronized(this) {
                    wait(3);
                    sendstr = rsaOBJ.Encrypt(aesOBJ.secretkeySpec);
                    System.out.printf("Encrypted AES Key : " + sendstr);
                    sendwriter.println(sendstr);
                    sendwriter.flush(); ////////////4
                }
                
                while(true) {
                    sendstr = tmpbuf.readLine();
                    
                    timestamp = new SimpleDateFormat("[yyyy/MM/dd HH:mm:ss]").format(new Date());
                    sendstr = "\"" + sendstr + "\"" + timestamp;
                    
                    sendstr = aesOBJ.Encrypt(sendstr);
                    sendwriter.println(sendstr);
                    sendwriter.flush();
                }
                
            } catch(IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        public void setSocket(Socket sock) {
            this.sock = sock;
        }
    }
    
    class AES {        
        SecretKey secretkey;
        String stringSecretkey;
        SecretKeySpec secretkeySpec;
        IvParameterSpec iv;

        public void createSecretkey() throws Exception {				
            // produce aes secrete key
            System.out.printf("Creating AES 256 Key");
            KeyGenerator keygenerator = KeyGenerator.getInstance("AES");
            SecureRandom randomnumber = SecureRandom.getInstance("SHA1PRNG");
            keygenerator.init(256, randomnumber);
            this.secretkey = keygenerator.generateKey();
            secretkeySpec = new SecretKeySpec(this.secretkey.getEncoded(), "AES");
            stringSecretkey = Base64.getEncoder().encodeToString(this.secretkey.getEncoded());
            System.out.println("AES 256 Key : " + stringSecretkey);
            
            // produce IV
            byte[] ivBytes = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15};
            this.iv = new IvParameterSpec(ivBytes);
            System.out.println("IV : " + this.iv);
        }
        
        public String Encrypt(String plainStr) throws Exception {
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, secretkeySpec, this.iv);
            byte[] encryptedByte = cipher.doFinal(plainStr.getBytes());
            String encryptedStr = new String(encryptedByte, "utf-8");
            
            return encryptedStr;
        }
        
        public String Decrpyt(String encryptedStr) throws Exception {
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretkeySpec, this.iv);
            byte[] decryptedByte = cipher.doFinal(encryptedStr.getBytes());
            String decryptedStr = new String(decryptedByte, "utf-8");
            
            return decryptedStr;
        }
    }

    class RSA{
		Key publickey;
		String strPublickey;
        
		public String Encrypt (String plainStr) throws Exception {
			Cipher cipher = Cipher.getInstance("RSA");
			cipher.init(Cipher.ENCRYPT_MODE, this.publickey);
			byte[] plainByte = cipher.doFinal(plainStr.getBytes());
			String encryptedStr = Base64.getEncoder().encodeToString(plainByte);
			
			return encryptedStr;
        }
	}
}