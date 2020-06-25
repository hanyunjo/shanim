
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.InputStreamReader;
import java.net.Socket;
import java.net.ServerSocket;

import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PublicKey;
import java.security.PrivateKey;

import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.spec.IvParameterSpec;

import java.util.Base64;
import java.util.Date;
import java.text.SimpleDateFormat;

/*public class Server {
	public static void main(String[] args) throws Exception {
		try {
			ServerSocket ser_sock = new ServerSocket(7777);
			//1. Create RSA Key Pair(2048bit)
			System.out.println("Creating RSA key pair");*/
			
			/*KeyPairGenerator keypairgenerator = KeyPairGenerator.getInstance("RSA");
			keypairgenerator.initialize(2048);
			KeyPair keypair = keypairgenerator.genKeyPair();
			PublicKey publickey = keypair.getPublic();
			PrivateKey privatekey = keypair.getPrivate();
			System.out.println("PublicKey : " + Base64.getEncoder().encodeToString(publickey.getEncoded()));
			System.out.println("PrivateKey : " + Base64.getEncoder().encodeToString(privatekey.getEncoded()));
			
			/*Socket cli_sock = ser_sock.accept();
			BufferedReader receivebuf = new BufferedReader(new InputStreamReader(cli_sock.getInputStream()));
			PrintWriter sendWriter = new PrintWriter(cli_sock.getOutputStream());
			String receivestr;*/
			
			// send publickey
			/*sendWriter.println(Base64.getEncoder().encodeToString(publickey.getEncoded()));
			sendWriter.flush();*/
			
			// receive privatekey and iv
			receivestr = receivebuf.readLine();
			System.out.println("Received AES key : " + receivestr);
			SecretKey secretkey = new SecretKeySpec(rsaDecrypt(privatekey, Base64.getDecoder().decode(receivestr)), 0, rsaDecrypt(privatekey, Base64.getDecoder().decode(receivestr)).length, "AES");
			System.out.println("Decrypted AES key : "+ Base64.getEncoder().encodeToString(secretkey.getEncoded()));
			
			receivestr = receivebuf.readLine();
			System.out.println("Received IV : " + receivestr);
			IvParameterSpec iv = new IvParameterSpec(rsaDecrypt(privatekey, Base64.getDecoder().decode(receivestr)));
			System.out.println("Decrypted IV : "+ Base64.getEncoder().encodeToString(iv.getIV()));			
			
			// start send & receive thread
			Receivethread rec_thread = new Receivethread(ser_sock, cli_sock, receivebuf, secretkey, iv);
			Sendthread sen_thread = new Sendthread(secretkey, iv, sendWriter);
			rec_thread.start();
			sen_thread.start();	
		} catch(IOException e) {
			e.printStackTrace();
		}
    }
    
    public static byte[] rsaDecrypt (PrivateKey privateKey, byte[] encryptedByte) throws Exception {
        Cipher cipher = Cipher.getInstance("RSA");
        cipher.init(Cipher.DECRYPT_MODE, privateKey);
        byte[] plainByte = cipher.doFinal(encryptedByte);
        return plainByte;
    }
}

/*class Receivethread extends Thread{
	private ServerSocket ser_sock;
	private Socket cli_sock;
	private BufferedReader receivebuf;
	private SecretKey secretkey;
	private IvParameterSpec iv;

	public Receivethread(ServerSocket ser_sock, Socket cli_sock, BufferedReader receivebuf, SecretKey secretkey, IvParameterSpec iv){
		this.ser_sock = ser_sock;
		this.cli_sock = cli_sock;
		this.receivebuf = receivebuf;
		this.secretkey = secretkey;
		this.iv = iv;
	}
	
	@Override
	public void run() {
		try {
			//BufferedReader receivebuf = new BufferedReader(new InputStreamReader(this.cli_sock.getInputStream()));
			String receivestr, decryptedstr;*/
			
			while(true) {
				/*receivestr = receivebuf.readLine();
				
				if(receivestr == null) {
					break;
				}*/
				
				decryptedstr = new String(aesDecrypt(this.secretkey, this.iv, receivestr.getBytes("UTF-8")), "UTF-8");
				// 4-1. Print data and Print decrypted data 
				System.out.println("Received : " + decryptedstr);
				System.out.println("Encrypted Message : " + receivestr);

				
				/*if(decryptedstr.substring(1,5).equals("exit")) {
					break;
				}*/
			}
			
			/*receivebuf.close();
			System.out.println("Connection closed");
			this.ser_sock.close();
            this.cli_sock.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}*/
	
    public static byte[] aesDecrypt(SecretKey secretkey, IvParameterSpec iv, byte[] encryptedByte) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.DECRYPT_MODE, secretkey, iv);
        byte[] base64Byte = Base64.getDecoder().decode(encryptedByte);
        byte[] decryptedByte = cipher.doFinal(base64Byte);        
        return decryptedByte;
    }
}

/*class Sendthread extends Thread{
	private SecretKey secretkey;
	private IvParameterSpec iv;
	public PrintWriter sendWriter;
	
	public Sendthread(SecretKey secretkey, IvParameterSpec iv, PrintWriter sendWriter){
		this.secretkey = secretkey;
		this.iv = iv;
		this.sendWriter = sendWriter;
	}
	
	@Override
	public void run() {
		try {
			BufferedReader sendbuf = new BufferedReader(new InputStreamReader(System.in));
			String sendstr, timestamp;
			int i = 0;
			
			while(true) {
				sendstr = sendbuf.readLine();
				if(sendstr.equals("exit")) i = 1;
				
				// 4-2. Make timestamp and Send encrypted data 
				timestamp = new SimpleDateFormat(" [yyyy/MM/dd HH:mm:ss]").format(new Date());
				sendstr = "\"" + sendstr + "\"" + timestamp;*/
				
				sendstr = new String(Base64.getEncoder().encode(aesEncrypt(this.secretkey, this.iv, sendstr.getBytes("UTF-8"))));
				/*sendWriter.println(sendstr);
				sendWriter.flush();
				
				if(i == 1) 
					break;
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}			
	}*/
	
	 public static byte[] aesEncrypt(SecretKey secretkey, IvParameterSpec iv, byte[] plainByte) throws Exception {
		Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
	    cipher.init(Cipher.ENCRYPT_MODE, secretkey, iv);
	    byte[] encryptedByte = cipher.doFinal(plainByte);        
	    return encryptedByte;
	}
}