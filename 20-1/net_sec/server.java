
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

import java.security.Key;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.spec.RSAPrivateKeySpec;
import java.security.spec.RSAPublicKeySpec;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Base64.Decoder;
import java.util.Base64.Encoder;

public class Mainserver {
	RSA rsaOBJ;	
	AES aesOBJ;

	public static void main(String[] args) throws Exception {
		try {
			ServerSocket ser_sock = new ServerSocket(9506);			
			rsaOBJ = new RSA();
			aesOBJ = new AES();
			
			rsaOBJ.creatKeyPair(); // 1. Create RSA Key Pair(2048bit)
			
			Socket cli_sock = ser_sock.accept();

			Receivethread rec_thread = new Receivethread();
			rec_thread.setSocket(cli_sock);
			rec_thread.start();
			
			// send_thread
			try {
				BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(System.in));
				PrintWriter sendwriter = new PrintWriter(cli_sock.getOutputStream());
				String sendstr, timestamp;
				int out = 0;
				
				synchronized(rec_thread) { // 2. Send Pulickey
					sendstr = rsaOBJ.strPublickey;
					sendwriter.println(sendstr);
					sendwriter.flush();
					rec_thread.notify(); 
					rec_thread.wait(); ////////////1
				}
			
				while(true) {
					sendstr = tmpbuf.readLine();
					
					// 4-2. Make timestamp and Send encrypted data 
					timestamp = new SimpleDateFormat("[yyyy/MM/dd HH:mm:ss]").format(new Date());
					sendstr = "\"" + sendstr + "\"" + timestamp;
					
					sendstr = aesOBJ.encrypt(sendstr);
					sendwriter.println(sendstr);
					sendwriter.flush();
				}
				
			} catch(IOException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}				
			
			System.out.println("Connection closed");
			ser_sock.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}

	class Receivethread extends Thread{
		private Socket sock;
		
		@Override
		public void run() {
			super.run();
			try {
				BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(sock.getInputStream()));
				String receivestr;
				
				RSA rsaOBJ = new RSA();
				AES aesOBJ = new AES();
				
				rsaOBJ.setprivatekey(this.privatekey);
				
				synchronized(this) { // 3 Take encrypted AES key and print decrypted AES key 
					wait(3);
					receivestr = tmpbuf.readLine(); ////////////2
					System.out.println("Received AES key : " + receivestr);
					System.out.println("Decrypted AES key : "+ rsaOBJ.decrypt(receivestr));
					sha_data.setSecretKeySpec(rsaOBJ.decrypt(receivestr));
					sha_data.setIvParameterSpec(rsaOBJ.decrypt(receivestr));
					notify();
				}
				
				while(true) {
					receivestr = tmpbuf.readLine();
					
					if(receivestring != null){
						// 4-1. Print data and Print decrypted data 
						System.out.println("Received : " + receivestr);
						System.out.println("Encrypted Message : " + aesOBJ.encrypt(receivestr));
					}
					
					if(receivestr.substring(1,4).equals("exit")) {
						break;
					}
				}

				tmpbuf.close();
                sock.close();
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
	
	class RSA{
		Key publickey;
		String strPublickey;
		Key privatekey;
		String strPrivatekey;

		public void creatKeyPair() throws Exception {
			System.out.println("Creating RSA key pair");
			
			KeyPairGenerator keypairgenerator = KeyPairGenerator.getInstance("RSA");
			keypairgenerator.initialize(2048);
				
			KeyPair keypair = keypairgenerator.genKeyPair();
			this.publickey = keypair.getPublic();
			this.privatekey = keypair.getPrivate();
			
			this.strPublickey = Base64.getEncoder().encodeToString(this.publickey.getEncoded());
			this.strPrivatekey = Base64.getEncoder().encodeToString(this.privatekey.getEncoded());
			
			System.out.println("PublicKey : " + this.strPublickey);
			System.out.println("PrivateKey : " + this.strPrivatekey);	
		}
		
		public String encrypt (String plainStr) throws Exception {
			Cipher cipher = Cipher.getInstance("RSA");
			cipher.init(Cipher.ENCRYPT_MODE, this.publickey);
			byte[] plainByte = cipher.doFinal(plainStr.getBytes());
			String encryptedStr = Base64.getEncoder().encodeToString(plainByte);
			
			return encryptedStr;
		}
		 
		public String decrypt (String encryptedStr) throws Exception {
			Cipher cipher = Cipher.getInstance("RSA");
			cipher.init(Cipher.DECRYPT_MODE, this.privatekey);
			byte[] encryptedByte = Base64.getDecoder().decode(encryptedStr.getBytes());
			byte[] plainByte = cipher.doFinal(encryptedByte);
			String decryptedStr = new String(plainByte, "utf-8");
			
			return decryptedStr;
		}
	}
	
	class AES{	
		SecretKeySpec secretkeySpec;
		IvParameterSpec iv;

		public String Encrypt(String plainStr) throws Exception {
			Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
			cipher.init(Cipher.ENCRYPT_MODE, secretkeySpec, this.iv);
			byte[] encryptedByte = cipher.doFinal(plainStr.getBytes());
			String encryptedStr = new String(encryptedByte);
			
			return encryptedStr;
		}
		
		public String Decrpyt(String encryptedStr) throws Exception {
			Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
			cipher.init(Cipher.DECRYPT_MODE, secretkeySpec, this.iv);
			byte[] decryptedByte = cipher.doFinal(encryptedStr.getBytes());
			String decryptedStr = new String(decryptedByte);
			
			return decryptedStr;
		}
	}
}