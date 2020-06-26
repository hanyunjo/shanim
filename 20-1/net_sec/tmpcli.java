import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import javax.crypto.*;
import javax.crypto.spec.*;
import java.util.Base64;
import java.security.spec.X509EncodedKeySpec;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.security.KeyFactory;
import java.security.PublicKey;

/*public class Client {	
	public static void main(String[] args) throws Exception {		
		try {			
			Socket sock = new Socket("127.0.0.1", 7777);
			BufferedReader receivebuf = new BufferedReader(new InputStreamReader(sock.getInputStream()));
			PrintWriter sendWriter = new PrintWriter(sock.getOutputStream());
			String receivestr, sendstr, timestamp;
			
			// receive publickey
			receivestr = receivebuf.readLine();*/
			/*System.out.println("Received PublicKey : " + receivestr);
			KeyFactory keyfactory = KeyFactory.getInstance("RSA");
			PublicKey publickey = keyfactory.generatePublic(new X509EncodedKeySpec(Base64.getDecoder().decode(receivestr)));*/
			
			/*//produce secretkey
        	System.out.println("Creating AES 256 Key");
        	KeyGenerator keygenerator = KeyGenerator.getInstance("AES");
        	keygenerator.init(256);
        	SecretKey secretkey = keygenerator.generateKey();
        	
        	// produce IV
        	String ivStr = new String(secretkey.getEncoded(), "UTF-8").substring(0,16);
        	IvParameterSpec iv = new IvParameterSpec(ivStr.getBytes());
        	byte[] ivByte = iv.getIV();*/
			
        	/*// send encrypted secretkey and iv
        	sendstr = new String(rsaEncrypt(publickey, secretkey.getEncoded()), "UTF-8");
        	System.out.println("AES 256 Key : " + Base64.getEncoder().encodeToString(secretkey.getEncoded()));
//        	sendstr = Base64.getEncoder().encodeToString(rsaEncrypt(publickey, secretkey.getEncoded()));
        	System.out.println("Encrypted AES Key : " + sendstr);
        	sendWriter.println(sendstr);
        	sendWriter.flush();
        	
        	sendstr = new String(rsaEncrypt(publickey, ivByte), "UTF-8");
        	System.out.println("IV : " + Base64.getEncoder().encodeToString(ivByte));
        	//sendstr = Base64.getEncoder().encodeToString(rsaEncrypt(publickey, ivByte));
        	System.out.println("Encrypted IV : " + sendstr);
        	sendWriter.println(sendstr);
			sendWriter.flush();*/
			
			// start send & receive thread
			/*CliReceivethread rec_thread = new CliReceivethread(receivebuf, secretkey, iv);
			CliSendthread sen_thread = new CliSendthread(secretkey, iv, sendWriter);
			rec_thread.start();
			sen_thread.start();
			rec_thread.join();
			
			timestamp = new SimpleDateFormat(" [yyyy/MM/dd HH:mm:ss]").format(new Date());
            sendstr = "\"exit\"" + timestamp;
            sendstr = new String(Base64.getEncoder().encode(aesEncrypt(secretkey, iv, sendstr.getBytes("UTF-8"))));
            sendWriter.println(sendstr);
            sendWriter.flush();
            System.out.println("Connection closed");
            sendWriter.close();
			sock.close();
			System.exit(0);
		} catch (Exception e) {
            e.printStackTrace();
        } 
    }*/
 	
 	/*public static byte[] rsaEncrypt (PublicKey publickey, byte[] plainByte) throws Exception {
        Cipher cipher = Cipher.getInstance("RSA");
        cipher.init(Cipher.ENCRYPT_MODE, publickey);
        byte[] encryptedByte = cipher.doFinal(plainByte);
        return encryptedByte;
    }*/
 	
 	public static byte[] aesEncrypt(SecretKey secretkey, IvParameterSpec iv, byte[] plainByte) throws Exception {
		Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
	    cipher.init(Cipher.ENCRYPT_MODE, secretkey, iv);
	    byte[] encryptedByte = cipher.doFinal(plainByte);        
	    return encryptedByte;
	}
}

class CliReceivethread extends Thread{
	private BufferedReader receivebuf;
	private SecretKey secretkey;
	private IvParameterSpec iv;
	
	public CliReceivethread(BufferedReader receivebuf, SecretKey secretkey, IvParameterSpec iv){
		this.receivebuf = receivebuf;
		this.secretkey = secretkey;
		this.iv = iv;
	}
    
    @Override
    public void run() { 			
        try {
        	//BufferedReader receivebuf = new BufferedReader(new InputStreamReader(this.sock.getInputStream()));
        	String receivestr, decryptedstr;
			
			while(true) {
				receivestr = receivebuf.readLine();
				decryptedstr = new String(aesDecrypt(this.secretkey, this.iv, receivestr.getBytes("UTF-8")), "UTF-8");
				
				if(receivestr != null){
					// 4-1. Print data and Print decrypted data 
					System.out.println("Received : " + decryptedstr);
					System.out.println("Encrypted Message : " + receivestr);
				}
				
        		if(decryptedstr.substring(1,5).equals("exit")) {
        			break;
        		}
        	}
        } catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    public static byte[] aesDecrypt(SecretKey secretkey, IvParameterSpec iv, byte[] encryptedByte) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.DECRYPT_MODE, secretkey, iv);
        byte[] base64Byte = Base64.getDecoder().decode(encryptedByte);
        byte[] decryptedByte = cipher.doFinal(base64Byte);        
        return decryptedByte;
    }
}
        
class CliSendthread extends Thread{
	private SecretKey secretkey;
	private IvParameterSpec iv;
	public PrintWriter sendWriter;
	
	public CliSendthread(SecretKey secretkey, IvParameterSpec iv, PrintWriter sendWriter){
		this.secretkey = secretkey;
		this.iv = iv;
		this.sendWriter = sendWriter;
	}
    
    @Override
    public void run() {
        try {
            BufferedReader sendbuf = new BufferedReader(new InputStreamReader(System.in));
            String sendstr, timestamp;
            
            while(true) {
                sendstr = sendbuf.readLine();
                
                //4-2. Make timestamp and Send encrypted data
                timestamp = new SimpleDateFormat(" [yyyy/MM/dd HH:mm:ss]").format(new Date());
                sendstr = "\"" + sendstr + "\"" + timestamp;
                
                sendstr = new String(Base64.getEncoder().encode(aesEncrypt(this.secretkey, this.iv, sendstr.getBytes("UTF-8"))));
                sendWriter.println(sendstr);
                sendWriter.flush();
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static byte[] aesEncrypt(SecretKey secretkey, IvParameterSpec iv, byte[] plainByte) throws Exception {
		Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
	    cipher.init(Cipher.ENCRYPT_MODE, secretkey, iv);
	    byte[] encryptedByte = cipher.doFinal(plainByte);        
	    return encryptedByte;
	}
}
        