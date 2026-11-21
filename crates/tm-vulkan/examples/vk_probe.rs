//! Prints every Vulkan device and its compute capabilities.
//! Run: `cargo run -p tm-vulkan --example vk_probe`

fn main() {
    match tm_vulkan::VulkanContext::list_devices() {
        Ok(devs) => {
            for d in devs {
                println!("{d:#?}");
            }
        }
        Err(e) => eprintln!("no Vulkan: {e}"),
    }
}
