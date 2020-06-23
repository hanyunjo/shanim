
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

import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

import java.util.Date;
import java.util.Base64.Decoder;
import java.util.Base64.Encoder;
import java.text.SimpleDateFormat;

public class Mainserver {	
	public static void main(String[] args) throws Exception {
		try {
			ServerSocket ser_sock = new ServerSocket(9506);
			Socket cli_sock = ser_sock.accept();
			
			RSA rsaOBJ = new RSA();
			AES aesOBJ = new AES();
			SharedData sharedDATA = new SharedData();
			
			rsaOBJ.creatKeyPair(); // 1. Create RSA Key Pair(2048bit)
			
			Receivethread rec_thread = new Receivethread(rsaOBJ.getprivatekey(), sharedDATA);
			rec_thread.setSocket(cli_sock);
			rec_thread.start();
			
			try {
				BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(System.in));
				PrintWriter sendwriter = new PrintWriter(cli_sock.getOutputStream());
				String sendstr, timestamp;
				int out = 0;
				
				synchronized(rec_thread) { // 2. Send Pulickey
					sendstr = rsaOBJ.getStrPublickey();
					sendwriter.println(sendstr);
					sendwriter.flush();
					rec_thread.notify();
					rec_thread.wait();
					aesOBJ.setSecretKeySpec(sharedDATA.getSecretKeySpec());
					aesOBJ.setIvParameterSpec(sharedDATA.getIvParameterSpec());
				}
			
				while(true) {
					sendstr = tmpbuf.readLine();
					if(sendstr.equals("exit")) {
						out = 1;
					}
					
					// 4-2. Make timestamp and Send encrypted data 
					timestamp = new SimpleDateFormat("[yyyy/MM/dd HH:mm:ss]").format(new Date());
					sendstr = sendstr + timestamp;
					
					sendstr = aesOBJ.encrypt(sendstr);
					sendwriter.println(sendstr);
					sendwriter.flush();
					if(out == 1) {
						break;
					}
				}
				
			} catch(IOException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}				
			
			System.out.println("Connection closed");
			ser_sock.close();
			cli_sock.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}
}

class SharedData{
	private SecretKeySpec secretkeySpec;
	private IvParameterSpec iv;
	
	public void setSecretKeySpec(SecretKeySpec secretkeySpec) {
		this.secretkeySpec = secretkeySpec;
	}
	public void setIvParameterSpec(IvParameterSpec iv) {
		this.iv = iv;
	}
	public SecretKeySpec getSecretKeySpec() {
		return this.secretkeySpec;
	}
	public IvParameterSpec getIvParameterSpec() {
		return this.iv;
	}
}

class Receivethread extends Thread{
	private Socket sock;
	private Key privatekey;
	private SharedData sha_data;
	
	public Receivethread(Key para1, SharedData sha_data) {
		this.privatekey = para1;
		this.sha_data = sha_data;
	}
	
	@Override
	public void run() {
		super.run();
		try {
			BufferedReader tmpbuf = new BufferedReader(new InputStreamReader(System.in));
			String receivestr;
			
			RSA rsaOBJ = new RSA();
			AES aesOBJ = new AES();
			
			rsaOBJ.setprivatekey(this.privatekey);
			
			synchronized(this) { // 3 Take encrypted AES key and print decrypted AES key 
				wait(3);
				receivestr = tmpbuf.readLine();
				System.out.println("Received AES key : " + receivestr);
				System.out.println("Decrypted AES key : "+ rsaOBJ.decrypt(receivestr));
				sha_data.setSecretKeySpec(rsaOBJ.decrypt(receivestr));
				sha_data.setIvParameterSpec(rsaOBJ.decrypt(receivestr));
				notify();
			}
			
			while(true) {
				receivestr = tmpbuf.readLine();
				
				// 4-1. Print data and Print decrypted data 
				System.out.println("Received : " + receivestr);
				System.out.println("Encrypted Message : " + aesOBJ.encrypt(receivestr));
				
				if(receivestr.equals("exit")) {
					break;
				}
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

class RSA{
	private Key publickey;
	private Key privatekey;
	private String strPublickey;
	
	//RSA
	public void creatKeyPair() throws Exception {
		System.out.println("Creating RSA key pair");
		
		KeyPairGenerator keypairgenerator = KeyPairGenerator.getInstance("RSA");
		keypairgenerator.initialize(2048);
			
		KeyPair keypair = keypairgenerator.genKeyPair();
		this.publickey = keypair.getPublic();
		this.privatekey = keypair.getPrivate();
		
		this.strPublickey = Base64.getEncoder().encodeToString(this.publickey.getEncoded());
		String strPrivatekey = Base64.getEncoder().encodeToString(this.privatekey.getEncoded());
		
		System.out.println("PublicKey : " + this.strPublickey);
		System.out.println("PrivateKey : " + strPrivatekey);	
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
	
	public Key getpublickey() {
		return this.publickey;
	}
	
	public Key getprivatekey() {
		return this.privatekey;
	}
	
	public String getStrPublickey() {
		return this.strPublickey;
	}
	
	public void setpublickey(Key pub) {
		this.publickey = pub;
	}
	
	public void setprivatekey(Key pri) {
		this.privatekey = pri;
	}
}

class AES{
	private SecretKeySpec secretkeySpec;
	private IvParameterSpec iv;

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
		
	public void setSecretKeySpec(SecretKeySpec secretKeySpec) {
		this.secretkeySpec = secretKeySpec;
	}
	
	public void setIvParameterSpec(IvParameterSpec iv) {
		this.iv = iv;
	}
}